# SPDX-License-Identifier: MIT
"""Hydra-powered experiment runner for TradePulse analytics."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from configs.secrets import default_secret_loader
from configs.settings import (
    ExperimentSettings,
    load_experiment_settings,
    register_structured_configs,
)

register_structured_configs()

from core.indicators.entropy import delta_entropy, entropy
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci


@dataclass(slots=True)
class RunMetadata:
    """Container for metadata captured for reproducibility."""

    run_dir: Path
    original_cwd: Path
    timestamp_utc: str
    git_sha: str | None
    python_version: str
    environment: str
    random_seed: int


def configure_logging(level_name: str) -> None:
    """Configure root logging with the requested level."""

    numeric_level = logging.getLevelName(level_name.upper())
    if isinstance(numeric_level, str):  # logging returns a string when lookup fails
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def set_random_seeds(seed: int) -> None:
    """Set deterministic seeds for Python's random and NumPy generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def apply_reproducibility_settings(cfg: DictConfig) -> None:
    """Apply optional reproducibility tweaks defined in the configuration."""

    repro_cfg = cfg.get("reproducibility")
    if repro_cfg is None:
        return

    precision = repro_cfg.get("numpy_print_precision")
    if precision is not None:
        np.set_printoptions(precision=int(precision))


def _current_git_sha(cwd: Path) -> str | None:
    """Return the git SHA for the repository if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _extract_experiment_section(
    settings: ExperimentSettings | DictConfig,
) -> ExperimentSettings | DictConfig:
    """Return the experiment configuration for both typed and raw inputs."""

    if isinstance(settings, ExperimentSettings):
        return settings

    experiment_cfg = settings.get("experiment")
    if experiment_cfg is not None:
        return experiment_cfg
    return settings


def collect_run_metadata(
    run_dir: Path, original_cwd: Path, settings: ExperimentSettings | DictConfig
) -> RunMetadata:
    """Collect metadata that allows reproducing the current experiment run."""

    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = _current_git_sha(original_cwd)
    experiment_cfg = _extract_experiment_section(settings)
    if isinstance(experiment_cfg, ExperimentSettings):
        environment = experiment_cfg.name
        seed = int(experiment_cfg.random_seed)
    else:
        environment = str(experiment_cfg.get("name", "")) or "default"
        seed = int(experiment_cfg.get("random_seed", 0))
    return RunMetadata(
        run_dir=run_dir,
        original_cwd=original_cwd,
        timestamp_utc=timestamp,
        git_sha=git_sha,
        python_version=platform.python_version(),
        environment=environment,
        random_seed=seed,
    )


def _write_metadata(metadata: RunMetadata) -> None:
    """Persist metadata to the Hydra run directory."""

    payload: dict[str, Any] = asdict(metadata)
    payload["run_dir"] = str(metadata.run_dir)
    payload["original_cwd"] = str(metadata.original_cwd)
    metadata_path = metadata.run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(settings: ExperimentSettings | DictConfig) -> dict[str, Any]:
    """Execute the analytics pipeline using configuration parameters."""

    logger = logging.getLogger("tradepulse.experiment")
    experiment_cfg = _extract_experiment_section(settings)

    if isinstance(experiment_cfg, ExperimentSettings):
        data_cfg = experiment_cfg.data
        analytics_cfg = experiment_cfg.analytics
        data_path_value = data_cfg.price_csv
        price_column = str(data_cfg.price_column)
        window = int(analytics_cfg.window)
        bins = int(analytics_cfg.bins)
        delta = float(analytics_cfg.delta)
    else:
        data_cfg = experiment_cfg.get("data")
        analytics_cfg = experiment_cfg.get("analytics")
        if data_cfg is None or analytics_cfg is None:
            raise ValueError("experiment configuration must define 'data' and 'analytics' sections")

        data_path_value = data_cfg.get("price_csv")
        price_column = str(data_cfg.get("price_column", "price"))
        window = int(analytics_cfg.get("window", 256))
        bins = int(analytics_cfg.get("bins", 48))
        delta = float(analytics_cfg.get("delta", 0.005))

    if data_path_value is None:
        raise ValueError("experiment configuration must set data.price_csv")

    data_path = Path(to_absolute_path(str(data_path_value)))
    if not data_path.exists():
        logger.warning("Data file %s does not exist; analytics step skipped.", data_path)
        return {"status": "missing-data", "path": str(data_path)}

    df = pd.read_csv(data_path)
    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' not found in dataset columns {list(df.columns)}"
        )

    prices = df[price_column].to_numpy()
    if len(prices) < window:
        raise ValueError(
            f"Not enough price observations ({len(prices)}) for window size {window}."
        )

    phases = compute_phase(prices)
    R = float(kuramoto_order(phases[-window:]))
    H = float(entropy(prices[-window:], bins=bins))
    dH = float(delta_entropy(prices, window=window))
    graph = build_price_graph(prices[-window:], delta=delta)
    kappa = float(mean_ricci(graph))

    summary = {
        "order_parameter": R,
        "entropy": H,
        "delta_entropy": dH,
        "mean_ricci": kappa,
        "window": window,
        "bins": bins,
        "delta": delta,
    }
    logger.info("Analytics summary computed: %s", summary)

    results_path = Path.cwd() / "results.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"status": "ok", "summary": summary, "results_path": str(results_path)}


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that orchestrates experiment execution."""

    secret_loader = default_secret_loader()
    experiment_settings = load_experiment_settings(cfg, secret_loader=secret_loader)

    configure_logging(experiment_settings.log_level)
    set_random_seeds(int(experiment_settings.random_seed))
    apply_reproducibility_settings(cfg)

    run_dir = Path.cwd()
    original_cwd = Path(get_original_cwd())
    metadata = collect_run_metadata(run_dir, original_cwd, experiment_settings)

    sanitized = experiment_settings.model_dump(mode="json")
    if "database" in sanitized:
        sanitized_database = dict(sanitized["database"])
        sanitized_database["uri"] = "<redacted>"
        sanitized_database.pop("password", None)
        sanitized["database"] = sanitized_database
    logging.getLogger(__name__).info(
        "Running with configuration:\n%s", json.dumps(sanitized, indent=2)
    )

    results = run_pipeline(experiment_settings)
    _write_metadata(metadata)

    results_path = run_dir / "pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
