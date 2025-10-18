# SPDX-License-Identifier: MIT
"""Hydra-powered experiment runner for TradePulse analytics."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
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

from core.config.hydra_profiles import (
    ExperimentProfileError,
    ExperimentProfileRegistry,
    validate_experiment_profile,
)
from core.indicators.entropy import delta_entropy, entropy
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci

REDACTED_PLACEHOLDER = "***REDACTED***"
SENSITIVE_TOKENS = (
    "password",
    "secret",
    "token",
    "apikey",
    "api_key",
    "credential",
    "auth",
    "key",
)


_CAMELCASE_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def _iter_key_tokens(key: str) -> list[str]:
    """Return normalized tokens extracted from a configuration key."""

    normalized = _CAMELCASE_BOUNDARY.sub(r"\1_\2", key)
    normalized = normalized.lower()
    return [token for token in re.split(r"[^0-9a-z]+", normalized) if token]


def _is_sensitive_key(key: str) -> bool:
    """Return ``True`` when the provided configuration key is sensitive."""

    return any(token in SENSITIVE_TOKENS for token in _iter_key_tokens(key))


def _redact_sensitive_data(data: Any) -> Any:
    """Return a copy of ``data`` with sensitive keys masked."""

    if isinstance(data, dict):
        return {
            key: (
                REDACTED_PLACEHOLDER
                if _is_sensitive_key(str(key))
                else _redact_sensitive_data(value)
            )
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_redact_sensitive_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_redact_sensitive_data(item) for item in data)
    return data


def _redacted_config_yaml(cfg: DictConfig) -> str:
    """Serialize ``cfg`` to YAML with sensitive values redacted."""

    container = OmegaConf.to_container(
        cfg,
        resolve=False,
        throw_on_missing=False,
    )
    redacted_container = _redact_sensitive_data(container)
    redacted_conf = OmegaConf.create(redacted_container)
    return OmegaConf.to_yaml(redacted_conf, resolve=False)


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


def collect_run_metadata(
    run_dir: Path, original_cwd: Path, cfg: DictConfig
) -> RunMetadata:
    """Collect metadata that allows reproducing the current experiment run."""

    timestamp = datetime.now(timezone.utc).isoformat()
    git_sha = _current_git_sha(original_cwd)
    environment = str(cfg.experiment.name)
    seed = int(cfg.experiment.random_seed)
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


def run_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Execute the analytics pipeline using configuration parameters."""

    logger = logging.getLogger("tradepulse.experiment")
    data_cfg = cfg.experiment.data
    analytics_cfg = cfg.experiment.analytics

    data_path = Path(to_absolute_path(str(data_cfg.price_csv)))
    if not data_path.exists():
        logger.warning(
            "Data file %s does not exist; analytics step skipped.", data_path
        )
        return {"status": "missing-data", "path": str(data_path)}

    df = pd.read_csv(data_path)
    price_column = str(data_cfg.price_column)
    if price_column not in df.columns:
        raise ValueError(
            f"Price column '{price_column}' not found in dataset columns {list(df.columns)}"
        )

    prices = df[price_column].to_numpy()
    window = int(analytics_cfg.window)
    bins = int(analytics_cfg.bins)
    delta = float(analytics_cfg.delta)

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

    registry: ExperimentProfileRegistry
    try:
        experiment = validate_experiment_profile(cfg)
        registry = ExperimentProfileRegistry.discover()
        registry.ensure(experiment.name)
    except ExperimentProfileError as exc:
        raise SystemExit(str(exc)) from exc

    configure_logging(experiment.log_level)
    set_random_seeds(experiment.random_seed)
    apply_reproducibility_settings(cfg)

    run_dir = Path.cwd()
    original_cwd = Path(get_original_cwd())
    metadata = collect_run_metadata(run_dir, original_cwd, cfg)

    logger = logging.getLogger(__name__)
    logger.info(
        "Using Hydra experiment profile '%s'. Available profiles: %s",
        experiment.name,
        ", ".join(registry.names()),
    )
    try:
        safe_yaml = _redacted_config_yaml(cfg)
    except Exception:
        logger.warning(
            "Unable to serialize configuration for safe logging; output suppressed.",
            exc_info=True,
        )
    else:
        logger.info("Running with configuration:\n%s", safe_yaml)

    results = run_pipeline(cfg)
    _write_metadata(metadata)

    results_path = run_dir / "pipeline_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
