"""TradePulse CLI exposing ingest/backtest/optimize/exec/report workflows."""

from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import click
import numpy as np
import pandas as pd

from core.config.cli_models import (
    BacktestConfig,
    ExecConfig,
    IngestConfig,
    OptimizeConfig,
    ReportConfig,
    StrategyConfig,
)
from core.config.template_manager import ConfigTemplateManager
from core.data.feature_catalog import FeatureCatalog
from core.data.versioning import DataVersionManager

DEFAULT_TEMPLATES_DIR = Path("configs/templates")


def _load_callable(entrypoint: str) -> Callable[..., Any]:
    module_name, _, attr_path = entrypoint.partition(":")
    if not attr_path:
        raise click.ClickException("Entrypoint must be in '<module>:<callable>' form")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_path.split("."):
        if not hasattr(target, part):
            raise click.ClickException(f"Entrypoint '{entrypoint}' is invalid")
        target = getattr(target, part)
    if not callable(target):
        raise click.ClickException(f"Entrypoint '{entrypoint}' does not reference a callable")
    return target  # type: ignore[return-value]


def _ensure_manager(ctx: click.Context, templates_dir: Path) -> ConfigTemplateManager:
    try:
        manager = ConfigTemplateManager(templates_dir)
    except FileNotFoundError as exc:  # pragma: no cover - user misconfiguration
        raise click.ClickException(str(exc)) from exc
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
        raise click.ClickException("Configuration does not define a data source")
    if data_cfg.kind not in {"csv", "parquet"}:
        raise click.ClickException(f"Unsupported data source '{data_cfg.kind}'")
    if not Path(data_cfg.path).exists():
        raise click.ClickException(f"Data source {data_cfg.path} does not exist")
    if data_cfg.kind == "csv":
        frame = pd.read_csv(data_cfg.path)
    else:
        frame = pd.read_parquet(data_cfg.path)
    if data_cfg.timestamp_field not in frame.columns:
        raise click.ClickException("Timestamp column missing from data source")
    if data_cfg.value_field not in frame.columns:
        raise click.ClickException("Value column missing from data source")
    frame = frame.sort_values(data_cfg.timestamp_field).reset_index(drop=True)
    return frame


def _resolve_strategy(strategy_cfg: StrategyConfig) -> Callable[[np.ndarray], np.ndarray]:
    fn = _load_callable(strategy_cfg.entrypoint)

    def _wrapped(prices: np.ndarray) -> np.ndarray:
        result = fn(prices, **strategy_cfg.parameters)
        return np.asarray(result, dtype=float)

    return _wrapped


def _write_frame(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".csv", ""}:
        frame.to_csv(destination, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        raise click.ClickException(f"Unsupported destination format '{suffix}'")


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
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to ingest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default ingest config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.pass_context
def ingest(ctx: click.Context, config: Path | None, generate_config: bool, output: Path | None) -> None:
    """Run data ingestion and register the produced artifact."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when generating a template")
        manager.render("ingest", output)
        click.echo(f"Template written to {output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    cfg = manager.load_config(config, IngestConfig)
    frame = _load_prices(cfg)
    _write_frame(frame, cfg.destination)

    catalog = FeatureCatalog(cfg.catalog)
    catalog.register(cfg.name, cfg.destination, config=cfg, lineage=[str(cfg.source.path)], metadata=cfg.metadata)

    version_mgr = DataVersionManager(cfg.versioning)
    version_mgr.snapshot(cfg.destination, metadata={"records": len(frame)})
    click.echo(f"Ingested {len(frame)} records to {cfg.destination}")


def _run_backtest(cfg: BacktestConfig) -> Dict[str, Any]:
    frame = _load_prices(cfg)
    prices = frame[cfg.data.value_field].to_numpy(dtype=float)
    strategy = _resolve_strategy(cfg.strategy)
    signals = strategy(prices)
    if signals.size != prices.size:
        raise click.ClickException("Strategy must return a signal for each price")
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


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to backtest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default backtest config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.pass_context
def backtest(ctx: click.Context, config: Path | None, generate_config: bool, output: Path | None) -> None:
    """Execute a simple vectorized backtest."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when generating a template")
        manager.render("backtest", output)
        click.echo(f"Template written to {output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    cfg = manager.load_config(config, BacktestConfig)
    result = _run_backtest(cfg)
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    catalog = FeatureCatalog(cfg.catalog)
    catalog.register(cfg.name, cfg.results_path, config=cfg, lineage=[str(cfg.data.path)], metadata=cfg.metadata)

    version_mgr = DataVersionManager(cfg.versioning)
    version_mgr.snapshot(cfg.results_path, metadata=result["stats"])
    click.echo(f"Backtest results written to {cfg.results_path}")


def _iterate_grid(search_space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space.keys())
    for key in keys:
        values = search_space[key]
        if not isinstance(values, Iterable):
            raise click.ClickException("Search space values must be iterable")
    for combo in itertools.product(*(search_space[key] for key in keys)):
        yield dict(zip(keys, combo))


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to optimization YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default optimize config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.pass_context
def optimize(ctx: click.Context, config: Path | None, generate_config: bool, output: Path | None) -> None:
    """Perform a brute-force search across a parameter grid."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when generating a template")
        manager.render("optimize", output)
        click.echo(f"Template written to {output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    cfg = manager.load_config(config, OptimizeConfig)
    backtest_cfg = BacktestConfig.model_validate(cfg.metadata.get("backtest")) if "backtest" in cfg.metadata else None
    if backtest_cfg is None:
        raise click.ClickException("Optimize config requires embedded backtest metadata")

    objective_fn = _load_callable(cfg.objective)
    trials: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params: Dict[str, Any] | None = None
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
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    version_mgr = DataVersionManager(cfg.versioning)
    version_mgr.snapshot(cfg.results_path, metadata={"trials": len(trials)})
    click.echo(f"Optimization results written to {cfg.results_path}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to exec YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default exec config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.pass_context
def exec(ctx: click.Context, config: Path | None, generate_config: bool, output: Path | None) -> None:  # noqa: A001
    """Evaluate the latest signal and persist it to disk."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when generating a template")
        manager.render("exec", output)
        click.echo(f"Template written to {output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    cfg = manager.load_config(config, ExecConfig)
    frame = _load_prices(cfg)
    prices = frame[cfg.data.value_field].to_numpy(dtype=float)
    strategy = _resolve_strategy(cfg.strategy)
    signals = strategy(prices)
    latest = float(signals[-1])
    result = {"latest_signal": latest, "count": int(signals.size)}
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    catalog = FeatureCatalog(cfg.catalog)
    catalog.register(cfg.name, cfg.results_path, config=cfg, lineage=[str(cfg.data.path)], metadata=cfg.metadata)

    version_mgr = DataVersionManager(cfg.versioning)
    version_mgr.snapshot(cfg.results_path, metadata=result)
    click.echo(f"Latest signal {latest} written to {cfg.results_path}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to report YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default report config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.pass_context
def report(ctx: click.Context, config: Path | None, generate_config: bool, output: Path | None) -> None:
    """Aggregate JSON artifacts into a markdown summary."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when generating a template")
        manager.render("report", output)
        click.echo(f"Template written to {output}")
        return
    if config is None:
        raise click.UsageError("--config is required when not generating a template")

    cfg = manager.load_config(config, ReportConfig)
    sections = []
    for artifact in cfg.inputs:
        path = Path(artifact)
        if not path.exists():
            raise click.ClickException(f"Report input {path} does not exist")
        sections.append(f"### {path.stem}\n``\n{path.read_text(encoding='utf-8').strip()}\n``")
    report_text = "\n\n".join(sections)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(report_text, encoding="utf-8")

    version_mgr = DataVersionManager(cfg.versioning)
    version_mgr.snapshot(cfg.output_path, metadata={"sections": len(sections)})
    click.echo(f"Report written to {cfg.output_path}")
