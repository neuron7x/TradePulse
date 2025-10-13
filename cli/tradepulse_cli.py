"""TradePulse CLI exposing ingest/materialize/backtest/train/serve/report workflows."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import io
import json
import sys
import time
from dataclasses import asdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Type

import click
import numpy as np
import pandas as pd

from core.utils.dataframe_io import (
    MissingParquetDependencyError,
    dataframe_to_parquet_bytes,
    read_dataframe,
)

from core.config.cli_models import (
    BacktestConfig,
    ExecConfig,
    IngestConfig,
    MaterializeConfig,
    OptimizeConfig,
    ReportConfig,
    ServeConfig,
    StrategyConfig,
    TrainConfig,
)
from core.config.template_manager import ConfigTemplateManager
from core.data.checkpoint_store import JsonCheckpointStore
from core.data.feature_catalog import FeatureCatalog
from core.data.feature_store import OnlineFeatureStore
from core.data.materialization import StreamMaterializer
from core.data.versioning import DataVersionManager
from core.neuro.calibration import CalibConfig, calibrate_random
from core.neuro.amm import AdaptiveMarketMind, AMMConfig
from core.reporting import generate_markdown_report, render_markdown_to_html, render_markdown_to_pdf

DEFAULT_TEMPLATES_DIR = Path("configs/templates")


class CLIError(click.ClickException):
    """Base class for typed CLI failures with deterministic exit codes."""

    exit_code = 1


class ConfigError(CLIError):
    exit_code = 2


class ArtifactError(CLIError):
    exit_code = 3


class ComputeError(CLIError):
    exit_code = 4


@contextmanager
def step_logger(command: str, name: str) -> Iterator[None]:
    """Context manager emitting deterministic start/stop step logs."""

    click.echo(f"[{command}] ▶ {name}")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        duration = time.perf_counter() - start
        click.echo(f"[{command}] ✖ {name} ({duration:.2f}s)", err=True)
        raise
    else:
        duration = time.perf_counter() - start
        click.echo(f"[{command}] ✓ {name} ({duration:.2f}s)")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _existing_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return _hash_bytes(path.read_bytes())


def _write_bytes(destination: Path, payload: bytes, *, command: str) -> Tuple[str, bool]:
    existing = _existing_digest(destination)
    digest = _hash_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing == digest:
        click.echo(f"[{command}] • {destination} unchanged (sha256={digest})")
        return digest, False
    destination.write_bytes(payload)
    click.echo(f"[{command}] • wrote {destination} (sha256={digest})")
    return digest, True


def _load_callable(entrypoint: str) -> Callable[..., Any]:
    module_name, _, attr_path = entrypoint.partition(":")
    if not attr_path:
        raise ConfigError("Entrypoint must be in '<module>:<callable>' form")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_path.split("."):
        if not hasattr(target, part):
            raise ConfigError(f"Entrypoint '{entrypoint}' is invalid")
        target = getattr(target, part)
    if not callable(target):
        raise ConfigError(f"Entrypoint '{entrypoint}' does not reference a callable")
    return target  # type: ignore[return-value]


def _ensure_manager(ctx: click.Context, templates_dir: Path) -> ConfigTemplateManager:
    try:
        manager = ConfigTemplateManager(templates_dir)
    except FileNotFoundError as exc:  # pragma: no cover - user misconfiguration
        raise ConfigError(str(exc)) from exc
    ctx.ensure_object(dict)
    ctx.obj["manager"] = manager
    return manager


def _get_manager(ctx: click.Context) -> ConfigTemplateManager:
    manager = ctx.obj.get("manager")
    if manager is None:
        manager = _ensure_manager(ctx, DEFAULT_TEMPLATES_DIR)
    return manager


def _load_prices(cfg: IngestConfig | BacktestConfig | ExecConfig) -> pd.DataFrame:
    data_cfg = getattr(cfg, "source", None)
    if data_cfg is None:
        data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        raise ConfigError("Configuration does not define a data source")
    frame = _read_source_frame(data_cfg)
    if data_cfg.timestamp_field not in frame.columns:
        raise ConfigError("Timestamp column missing from data source")
    if data_cfg.value_field not in frame.columns:
        raise ConfigError("Value column missing from data source")
    frame = frame.sort_values(data_cfg.timestamp_field).reset_index(drop=True)
    return frame


def _resolve_strategy(strategy_cfg: StrategyConfig) -> Callable[[np.ndarray], np.ndarray]:
    fn = _load_callable(strategy_cfg.entrypoint)

    def _wrapped(prices: np.ndarray) -> np.ndarray:
        result = fn(prices, **strategy_cfg.parameters)
        return np.asarray(result, dtype=float)

    return _wrapped


def _write_frame(frame: pd.DataFrame, destination: Path, *, command: str = "cli") -> str:
    suffix = destination.suffix.lower()
    if suffix in {".csv", ""}:
        payload = frame.to_csv(index=False).encode("utf-8")
    elif suffix == ".parquet":
        try:
            payload = dataframe_to_parquet_bytes(frame, index=False)
        except MissingParquetDependencyError as exc:
            raise ConfigError(
                "Writing parquet outputs requires either pyarrow or polars. Install the 'tradepulse[feature_store]' extra."
            ) from exc
    else:
        raise ConfigError(f"Unsupported destination format '{suffix}'")
    digest, _ = _write_bytes(destination, payload, command=command)
    return digest


def _write_json(destination: Path, payload: Dict[str, Any], *, command: str) -> str:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    digest, _ = _write_bytes(destination, data, command=command)
    return digest


def _write_text(destination: Path, text: str, *, command: str) -> str:
    digest, _ = _write_bytes(destination, text.encode("utf-8"), command=command)
    return digest


def _read_source_frame(source_cfg: Any, *, allow_json_fallback: bool = False) -> pd.DataFrame:
    if not hasattr(source_cfg, "kind") or not hasattr(source_cfg, "path"):
        raise ConfigError("Source configuration must define 'kind' and 'path'")
    if source_cfg.kind not in {"csv", "parquet"}:
        raise ConfigError(f"Unsupported data source '{source_cfg.kind}'")
    path = Path(source_cfg.path)
    if not path.exists():
        raise ArtifactError(f"Data source {path} does not exist")
    if source_cfg.kind == "csv":
        frame = pd.read_csv(path)
    else:
        try:
            frame = read_dataframe(path, allow_json_fallback=allow_json_fallback)
        except MissingParquetDependencyError as exc:
            raise ArtifactError(
                "Parquet sources require either pyarrow or polars. Install the 'tradepulse[feature_store]' extra."
            ) from exc
    return frame


def _resolve_feature_artifact_path(store_root: Path, feature_view: str) -> Path:
    safe_name = feature_view.replace("/", "__").replace(".", "__")
    base = Path(store_root) / safe_name
    for suffix in (".parquet", ".json"):
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise ArtifactError(f"No persisted artifact found for feature view '{feature_view}' at {base}.*")


@click.group()
@click.option(
    "--templates-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_TEMPLATES_DIR,
    help="Directory containing YAML configuration templates.",
)
@click.pass_context
def cli(ctx: click.Context, templates_dir: Path) -> None:
    """TradePulse orchestration CLI."""

    _ensure_manager(ctx, templates_dir)


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Generate shell completion snippet for the requested shell."""

    prog_name = Path(sys.argv[0]).name or "tradepulse_cli"
    env_prefix = prog_name.replace("-", "_").upper()
    env_var = f"{env_prefix}_COMPLETE"
    if shell == "bash":
        snippet = f'eval "$({env_var}=bash_source {prog_name})"'
    elif shell == "zsh":
        snippet = f'eval "$({env_var}=zsh_source {prog_name})"'
    else:  # fish
        snippet = f"eval (env {env_var}=fish_source {prog_name})"
    click.echo(f"# Add the following line to your {shell} configuration file")
    click.echo(snippet)


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to ingest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default ingest config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.pass_context
def ingest(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
) -> None:
    """Run data ingestion and register the produced artifact."""

    command = "ingest"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("ingest", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, IngestConfig)
    with step_logger(command, "load source data"):
        frame = _load_prices(cfg)
        record_count = len(frame)
    with step_logger(command, "persist dataset"):
        digest = _write_frame(frame, cfg.destination, command=command)
    with step_logger(command, "register catalog"):
        catalog = FeatureCatalog(cfg.catalog)
        entry = catalog.register(
            cfg.name,
            cfg.destination,
            config=cfg,
            lineage=[str(cfg.source.path)],
            metadata=cfg.metadata,
        )
        click.echo(f"[{command}] • catalog checksum={entry.checksum}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.destination, metadata={"records": record_count})
    click.echo(f"[{command}] completed records={record_count} dest={cfg.destination} sha256={digest}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to materialize YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default materialize config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def materialize(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Materialise features into the online feature store."""

    command = "materialize"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("materialize", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, MaterializeConfig)
    with step_logger(command, "load payload"):
        frame = _read_source_frame(cfg.source, allow_json_fallback=True)
    with step_logger(command, "materialize stream"):
        store = OnlineFeatureStore(cfg.store_root)
        try:
            checkpoint_store = JsonCheckpointStore(cfg.checkpoint_path)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc

        materializer = StreamMaterializer(
            lambda feature_view, batch: store.sync(feature_view, batch, mode="append", validate=True),
            checkpoint_store,
            microbatch_size=cfg.microbatch_size,
            dedup_keys=tuple(cfg.dedup_keys),
            backfill_loader=lambda feature_view: store.load(feature_view),
        )
        materializer.materialize(cfg.feature_view, frame)
        artifact_path = _resolve_feature_artifact_path(cfg.store_root, cfg.feature_view)
        digest = _existing_digest(artifact_path)
        if digest is None:
            raise ArtifactError(f"Failed to persist artifact for feature view '{cfg.feature_view}'")
        stored_frame = store.load(cfg.feature_view)
        row_count = int(stored_frame.shape[0])
    with step_logger(command, "register catalog"):
        catalog = FeatureCatalog(cfg.catalog)
        entry = catalog.register(
            cfg.name,
            artifact_path,
            config=cfg,
            lineage=[str(cfg.source.path)],
            metadata=cfg.metadata,
        )
        click.echo(f"[{command}] • catalog checksum={entry.checksum}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(artifact_path, metadata={"rows": row_count})
    _emit_materialize_output(cfg, row_count, artifact_path, output_format, command=command)
    click.echo(
        f"[{command}] completed feature_view={cfg.feature_view} rows={row_count} artifact={artifact_path} sha256={digest}"
    )


def _run_backtest(cfg: BacktestConfig) -> Dict[str, Any]:
    frame = _load_prices(cfg)
    prices = frame[cfg.data.value_field].to_numpy(dtype=float)
    strategy = _resolve_strategy(cfg.strategy)
    signals = strategy(prices)
    if signals.size != prices.size:
        raise ComputeError("Strategy must return a signal for each price")
    prev_prices = np.concatenate(([prices[0]], prices[:-1]))
    denom = np.where(prev_prices == 0, 1.0, prev_prices)
    returns = (prices - prev_prices) / denom
    pnl = returns * signals
    equity = cfg.execution.starting_cash * (1 + pnl.cumsum())
    stats = {
        "total_return": float(pnl.sum()),
        "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1.0)),
        "trades": int(np.count_nonzero(np.diff(signals) != 0)),
    }
    return {"stats": stats, "signals": signals.tolist(), "returns": pnl.tolist()}


def _emit_backtest_output(
    cfg: BacktestConfig,
    result: Dict[str, Any],
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("metric | value")
        click.echo("------ | -----")
        for metric, value in result["stats"].items():
            click.echo(f"{metric} | {value}")
        return
    if output_format == "jsonl":
        for metric, value in result["stats"].items():
            click.echo(json.dumps({"metric": metric, "value": value}))
        return
    if output_format == "parquet":
        frame = pd.DataFrame(
            {
                "step": np.arange(len(result["signals"])),
                "signal": result["signals"],
                "return": result["returns"],
            }
        )
        parquet_path = cfg.results_path.with_suffix(".parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


def _emit_materialize_output(
    cfg: MaterializeConfig,
    row_count: int,
    artifact_path: Path,
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("metric | value")
        click.echo("------ | -----")
        click.echo(f"feature_view | {cfg.feature_view}")
        click.echo(f"rows | {row_count}")
        click.echo(f"artifact | {artifact_path}")
        return
    if output_format == "jsonl":
        click.echo(json.dumps({"metric": "feature_view", "value": cfg.feature_view}))
        click.echo(json.dumps({"metric": "rows", "value": row_count}))
        click.echo(json.dumps({"metric": "artifact", "value": str(artifact_path)}))
        return
    if output_format == "parquet":
        frame = pd.DataFrame(
            [
                {"metric": "feature_view", "value": cfg.feature_view},
                {"metric": "rows", "value": row_count},
                {"metric": "artifact", "value": str(artifact_path)},
            ]
        )
        parquet_path = artifact_path.with_suffix(".summary.parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to backtest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default backtest config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Execute a simple vectorized backtest."""

    command = "backtest"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("backtest", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, BacktestConfig)
    with step_logger(command, "run backtest"):
        result = _run_backtest(cfg)
    with step_logger(command, "persist results"):
        digest = _write_json(cfg.results_path, result, command=command)
    with step_logger(command, "register catalog"):
        catalog = FeatureCatalog(cfg.catalog)
        entry = catalog.register(
            cfg.name,
            cfg.results_path,
            config=cfg,
            lineage=[str(cfg.data.path)],
            metadata=cfg.metadata,
        )
        click.echo(f"[{command}] • catalog checksum={entry.checksum}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.results_path, metadata=result["stats"])
    _emit_backtest_output(cfg, result, output_format, command=command)
    click.echo(f"[{command}] completed stats={json.dumps(result['stats'], sort_keys=True)} sha256={digest}")


def _iterate_grid(search_space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space.keys())
    for key in keys:
        values = search_space[key]
        if not isinstance(values, Iterable):
            raise ConfigError("Search space values must be iterable")
    for combo in itertools.product(*(search_space[key] for key in keys)):
        yield dict(zip(keys, combo))


def _emit_optimize_output(
    cfg: OptimizeConfig,
    payload: Dict[str, Any],
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("metric | value")
        click.echo("------ | -----")
        click.echo(f"best_score | {payload['best_score']}")
        if payload["best_params"]:
            for key, value in payload["best_params"].items():
                click.echo(f"param:{key} | {value}")
        click.echo(f"trials | {len(payload['trials'])}")
        return
    if output_format == "jsonl":
        click.echo(json.dumps({"metric": "best_score", "value": payload["best_score"]}))
        if payload["best_params"]:
            click.echo(json.dumps({"metric": "best_params", "value": payload["best_params"]}))
        for trial in payload["trials"]:
            click.echo(json.dumps({"metric": "trial", "value": trial}))
        return
    if output_format == "parquet":
        frame = pd.json_normalize(payload["trials"])
        parquet_path = cfg.results_path.with_suffix(".parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to optimization YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default optimize config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def optimize(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Perform a brute-force search across a parameter grid."""

    command = "optimize"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("optimize", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, OptimizeConfig)
    backtest_cfg = BacktestConfig.model_validate(cfg.metadata.get("backtest")) if "backtest" in cfg.metadata else None
    if backtest_cfg is None:
        raise ConfigError("Optimize config requires embedded backtest metadata")

    with step_logger(command, "load objective"):
        objective_fn = _load_callable(cfg.objective)
    trials: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params: Dict[str, Any] | None = None
    with step_logger(command, "grid search"):
        for params in _iterate_grid(cfg.search_space):
            trial_cfg = backtest_cfg.model_copy(deep=True)
            trial_cfg.strategy.parameters.update(params)
            trial_result = _run_backtest(trial_cfg)
            returns = np.asarray(trial_result["returns"], dtype=float)
            score = float(objective_fn(returns))
            trials.append({"params": params, "score": score, "stats": trial_result["stats"]})
            if score > best_score:
                best_score = score
                best_params = params

    payload = {"best_score": best_score, "best_params": best_params, "trials": trials}
    with step_logger(command, "persist results"):
        digest = _write_json(cfg.results_path, payload, command=command)
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.results_path, metadata={"trials": len(trials)})
    _emit_optimize_output(cfg, payload, output_format, command=command)
    click.echo(f"[{command}] completed trials={len(trials)} best_score={best_score} sha256={digest}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to train YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default train config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def train(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Calibrate the Adaptive Market Mind model on historical data."""

    command = "train"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("train", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, TrainConfig)
    with step_logger(command, "load dataset"):
        frame = _read_source_frame(cfg.data, allow_json_fallback=True)
        required = {cfg.data.signal_field, cfg.data.reward_field, cfg.data.kappa_field}
        missing = required - set(frame.columns)
        if missing:
            raise ConfigError(f"Training dataset missing columns: {sorted(missing)}")
        if frame.empty:
            raise ConfigError("Training dataset must contain at least one row")
        signal = frame[cfg.data.signal_field].to_numpy(dtype=float)
        reward = frame[cfg.data.reward_field].to_numpy(dtype=float)
        kappa = frame[cfg.data.kappa_field].to_numpy(dtype=float)
    with step_logger(command, "calibrate model"):
        calib_cfg = CalibConfig(**cfg.calibration.model_dump())
        best = calibrate_random(signal, reward, kappa, calib_cfg)
        score = _evaluate_calibration(best, signal, reward, kappa)
        result = {
            "best_params": asdict(best),
            "score": score,
            "records": int(frame.shape[0]),
            "trials": cfg.calibration.iters,
        }
    with step_logger(command, "persist results"):
        digest = _write_json(cfg.results_path, result, command=command)
    with step_logger(command, "register catalog"):
        catalog = FeatureCatalog(cfg.catalog)
        entry = catalog.register(
            cfg.name,
            cfg.results_path,
            config=cfg,
            lineage=[str(cfg.data.path)],
            metadata=cfg.metadata,
        )
        click.echo(f"[{command}] • catalog checksum={entry.checksum}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.results_path, metadata={"score": score, "records": int(frame.shape[0])})
    _emit_train_output(cfg, result, output_format, command=command)
    click.echo(f"[{command}] completed score={score} trials={cfg.calibration.iters} sha256={digest}")


def _evaluate_calibration(
    cfg: AMMConfig,
    signal: np.ndarray,
    reward: np.ndarray,
    kappa: np.ndarray,
) -> float:
    if signal.size == 0:
        return 0.0
    amm = AdaptiveMarketMind(cfg)
    pulses = np.zeros(signal.size, dtype=float)
    precision = np.zeros(signal.size, dtype=float)
    errors = np.zeros(signal.size, dtype=float)
    for idx in range(signal.size):
        output = amm.update(float(signal[idx]), float(reward[idx]), float(kappa[idx]))
        pulses[idx] = float(output["amm_pulse"])
        precision[idx] = float(output["amm_precision"])
        errors[idx] = abs(float(output["pe"]))
    try:
        corr = float(np.corrcoef(errors, pulses)[0, 1])
    except (FloatingPointError, ValueError):  # pragma: no cover - defensive
        corr = 0.0
    avg_precision = float(np.mean(np.clip(precision, 0.01, 100.0))) if precision.size else 0.0
    score = corr * avg_precision
    if np.isnan(score):
        return 0.0
    return score


def _emit_train_output(
    cfg: TrainConfig,
    result: Dict[str, Any],
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("metric | value")
        click.echo("------ | -----")
        click.echo(f"score | {result['score']}")
        click.echo(f"records | {result['records']}")
        click.echo(f"trials | {result['trials']}")
        for key, value in result["best_params"].items():
            click.echo(f"param:{key} | {value}")
        return
    if output_format == "jsonl":
        click.echo(json.dumps({"metric": "score", "value": result["score"]}))
        click.echo(json.dumps({"metric": "records", "value": result["records"]}))
        click.echo(json.dumps({"metric": "trials", "value": result["trials"]}))
        click.echo(json.dumps({"metric": "best_params", "value": result["best_params"]}))
        return
    if output_format == "parquet":
        rows = [
            {"metric": "score", "value": result["score"]},
            {"metric": "records", "value": result["records"]},
            {"metric": "trials", "value": result["trials"]},
        ]
        rows.extend({"metric": f"param:{key}", "value": value} for key, value in result["best_params"].items())
        frame = pd.DataFrame(rows)
        parquet_path = cfg.results_path.with_suffix(".parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


def _emit_signal_output(
    cfg: ExecConfig,
    result: Dict[str, Any],
    signals: np.ndarray,
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("metric | value")
        click.echo("------ | -----")
        for metric, value in result.items():
            click.echo(f"{metric} | {value}")
        return
    if output_format == "jsonl":
        for metric, value in result.items():
            click.echo(json.dumps({"metric": metric, "value": value}))
        return
    if output_format == "parquet":
        frame = pd.DataFrame({"step": np.arange(signals.size), "signal": signals})
        parquet_path = cfg.results_path.with_suffix(".parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


def _run_signal_command(
    ctx: click.Context,
    *,
    command: str,
    template_name: str,
    config_path: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
    config_model: Type[ExecConfig],
) -> None:
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render(template_name, template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config_path is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config_path, config_model)
    with step_logger(command, "load data"):
        frame = _load_prices(cfg)
        prices = frame[cfg.data.value_field].to_numpy(dtype=float)
    with step_logger(command, "evaluate strategy"):
        strategy = _resolve_strategy(cfg.strategy)
        signals = strategy(prices)
    if signals.size == 0:
        raise ComputeError("Strategy must emit at least one signal")

    latest = float(signals[-1])
    result = {"latest_signal": latest, "count": int(signals.size)}
    with step_logger(command, "persist results"):
        digest = _write_json(cfg.results_path, result, command=command)
    with step_logger(command, "register catalog"):
        catalog = FeatureCatalog(cfg.catalog)
        entry = catalog.register(
            cfg.name,
            cfg.results_path,
            config=cfg,
            lineage=[str(cfg.data.path)],
            metadata=cfg.metadata,
        )
        click.echo(f"[{command}] • catalog checksum={entry.checksum}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.results_path, metadata=result)
    _emit_signal_output(cfg, result, signals, output_format, command=command)
    click.echo(f"[{command}] completed latest_signal={latest} sha256={digest}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to exec YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default exec config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def exec(  # noqa: A001
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Evaluate the latest signal and persist it to disk."""

    _run_signal_command(
        ctx,
        command="exec",
        template_name="exec",
        config_path=config,
        generate_config=generate_config,
        template_output=template_output,
        output_format=output_format,
        config_model=ExecConfig,
    )


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to serve YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default serve config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Produce the latest trading signal for downstream consumers."""

    _run_signal_command(
        ctx,
        command="serve",
        template_name="serve",
        config_path=config,
        generate_config=generate_config,
        template_output=template_output,
        output_format=output_format,
        config_model=ServeConfig,
    )


def _emit_report_output(
    cfg: ReportConfig,
    report_text: str,
    output_format: str | None,
    *,
    command: str,
) -> None:
    if output_format is None:
        return
    if output_format == "table":
        click.echo("section | source")
        click.echo("------- | ------")
        for idx, path in enumerate(cfg.inputs, start=1):
            click.echo(f"section-{idx} | {path}")
        click.echo(f"length | {len(report_text.splitlines())}")
        return
    if output_format == "jsonl":
        for idx, path in enumerate(cfg.inputs, start=1):
            click.echo(json.dumps({"section": idx, "source": str(path)}))
        click.echo(json.dumps({"metric": "line_count", "value": len(report_text.splitlines())}))
        return
    if output_format == "parquet":
        frame = pd.DataFrame({"section": range(1, len(cfg.inputs) + 1), "source": [str(p) for p in cfg.inputs]})
        parquet_path = cfg.output_path.with_suffix(".parquet")
        _write_frame(frame, parquet_path, command=command)
        return
    raise ConfigError(f"Unsupported output format '{output_format}'")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to report YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default report config template.")
@click.option(
    "--template-output",
    type=click.Path(path_type=Path),
    help="Destination for generated template.",
)
@click.option(
    "--output-format",
    "--output",
    type=click.Choice(["table", "jsonl", "parquet"]),
    help="Render results in the requested format in addition to the persisted artifact.",
)
@click.pass_context
def report(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    template_output: Path | None,
    output_format: str | None,
) -> None:
    """Aggregate JSON artifacts into a markdown summary."""

    command = "report"
    manager = _get_manager(ctx)
    if generate_config:
        if template_output is None:
            raise click.UsageError("--template-output must be provided when generating a template")
        manager.render("report", template_output)
        click.echo(f"[{command}] template written to {template_output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    with step_logger(command, "load config"):
        cfg = manager.load_config(config, ReportConfig)
    with step_logger(command, "generate markdown"):
        try:
            report_text = generate_markdown_report(cfg)
        except FileNotFoundError as exc:
            raise ArtifactError(str(exc)) from exc
    with step_logger(command, "persist markdown"):
        digest = _write_text(cfg.output_path, report_text, command=command)
    if cfg.html_output_path is not None:
        with step_logger(command, "render html"):
            render_markdown_to_html(report_text, cfg.html_output_path)
            html_digest = _existing_digest(cfg.html_output_path)
            if html_digest:
                click.echo(f"[{command}] • html sha256={html_digest}")
    if cfg.pdf_output_path is not None:
        with step_logger(command, "render pdf"):
            render_markdown_to_pdf(report_text, cfg.pdf_output_path)
            pdf_digest = _existing_digest(cfg.pdf_output_path)
            if pdf_digest:
                click.echo(f"[{command}] • pdf sha256={pdf_digest}")
    with step_logger(command, "snapshot version"):
        version_mgr = DataVersionManager(cfg.versioning)
        version_mgr.snapshot(cfg.output_path, metadata={"sections": len(cfg.inputs)})
    _emit_report_output(cfg, report_text, output_format, command=command)
    click.echo(f"[{command}] completed sections={len(cfg.inputs)} sha256={digest}")
