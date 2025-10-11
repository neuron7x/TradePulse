"""Unified TradePulse CLI exposing ingest/backtest/optimize/exec/report."""

from __future__ import annotations

import importlib
import inspect
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import click
import numpy as np
import pandas as pd

from backtest.event_driven import EventDrivenBacktestEngine, LatencyConfig, OrderBookConfig, SlippageConfig
from core.config import (
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
from core.data.ingestion import DataIngestor

DEFAULT_TEMPLATES_DIR = Path("configs/templates")


def _load_callable(entrypoint: str) -> Callable[..., Any]:
    module_name, attr_path = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_path.split("."):
        if not hasattr(target, part):
            raise click.ClickException(f"Entrypoint '{entrypoint}' is invalid; missing attribute '{part}'")
        target = getattr(target, part)
    if not callable(target):
        raise click.ClickException(f"Entrypoint '{entrypoint}' does not refer to a callable")
    return target


def _ensure_manager(ctx: click.Context, templates_dir: Path | None) -> ConfigTemplateManager:
    try:
        manager = ConfigTemplateManager(templates_dir or DEFAULT_TEMPLATES_DIR)
    except FileNotFoundError as exc:  # pragma: no cover - user configuration issue
        raise click.ClickException(str(exc)) from exc
    ctx.ensure_object(dict)
    ctx.obj["manager"] = manager
    return manager


def _get_manager(ctx: click.Context) -> ConfigTemplateManager:
    manager = ctx.obj.get("manager")
    if manager is None:
        manager = _ensure_manager(ctx, DEFAULT_TEMPLATES_DIR)
    return manager


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


def _handle_generate(manager: ConfigTemplateManager, template: str, destination: Path) -> None:
    manager.render(template, destination)
    click.echo(f"Generated {template} config at {destination}")


def _resolve_signal_callable(strategy: StrategyConfig) -> Callable[[np.ndarray], np.ndarray]:
    signal_fn = _load_callable(strategy.entrypoint)

    def _wrapper(prices: np.ndarray) -> np.ndarray:
        return np.asarray(signal_fn(prices, **strategy.parameters), dtype=float)

    return _wrapper


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to ingest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default ingest config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.option("--validate-only", is_flag=True, help="Validate configuration without running the job.")
@click.pass_context
def ingest(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    output: Path | None,
    validate_only: bool,
) -> None:
    """Run data ingestion with quality checks and catalog registration."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when using --generate-config")
        _handle_generate(manager, "ingest", output)
        return
    if config is None:
        raise click.UsageError("--config is required for ingestion")

    cfg = manager.load_config(config, IngestConfig)
    if validate_only:
        click.echo(f"Configuration {config} is valid.")
        return

    if cfg.source.kind != "csv":  # pragma: no cover - future work
        raise click.ClickException("Currently only CSV ingestion is supported via the CLI")
    if cfg.source.path is None:
        raise click.ClickException("Source path must be provided for CSV ingestion")

    ingestor = DataIngestor()
    records: List[Dict[str, Any]] = []
    required_fields = tuple(cfg.source.parameters.get("required_fields", ("ts", "price")))
    symbol = cfg.source.parameters.get("symbol", cfg.metadata.get("symbol", "UNKNOWN"))
    venue = cfg.source.parameters.get("venue", "CSV")

    def _collect(tick: Any) -> None:
        if hasattr(tick, "price"):
            payload = {
                "symbol": getattr(tick, "symbol", symbol),
                "venue": getattr(tick, "venue", venue),
                "instrument_type": getattr(getattr(tick, "instrument_type", "spot"), "value", "spot"),
                "timestamp": getattr(tick, "timestamp", datetime.now(tz=timezone.utc)).isoformat(),
                "price": float(getattr(tick, "price", 0.0)),
                "volume": float(getattr(tick, "volume", 0.0)),
            }
        else:  # pragma: no cover - defensive branch for dict payloads
            payload = dict(tick)
        records.append(payload)

    ingestor.historical_csv(
        str(cfg.source.path),
        _collect,
        required_fields=required_fields,
        symbol=symbol,
        venue=venue,
    )
    if not records:
        raise click.ClickException("Ingestion completed but produced no records")

    frame = pd.DataFrame(records)
    cfg.destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = cfg.destination.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(cfg.destination, index=False)
    elif suffix == ".csv":
        frame.to_csv(cfg.destination, index=False)
    else:
        cfg.destination.write_text(frame.to_json(orient="records"), encoding="utf-8")

    lineage = list(cfg.lineage)
    if cfg.source.path is not None:
        lineage.append(str(cfg.source.path))
    catalog = FeatureCatalog(cfg.catalog.path)
    catalog.register(
        cfg.name,
        cfg.destination,
        config=cfg,
        lineage=lineage,
        metadata=cfg.metadata,
    )
    versioner = DataVersionManager(cfg.versioning)
    versioner.snapshot(cfg.destination, push=bool(cfg.versioning.remote))
    click.echo(f"Ingested {len(records)} records into {cfg.destination}")


def _load_prices(cfg: BacktestConfig) -> tuple[np.ndarray, str]:
    if cfg.data.kind == "csv":
        if cfg.data.path is None:
            raise click.ClickException("CSV data source requires a path")
        params = cfg.data.parameters
        price_column = params.get("price_column", "close")
        parse_dates = bool(params.get("parse_dates", False))
        read_kwargs: Dict[str, Any] = {}
        if parse_dates and params.get("date_column"):
            read_kwargs["parse_dates"] = [params["date_column"]]
        frame = pd.read_csv(cfg.data.path, **read_kwargs)
        if price_column not in frame.columns:
            raise click.ClickException(f"Column '{price_column}' not present in {cfg.data.path}")
        series = frame[price_column].astype(float).to_numpy()
        symbol = params.get("symbol", cfg.metadata.get("symbol", "asset"))
        return series, symbol
    if cfg.data.kind == "parquet":  # pragma: no cover - optional path
        if cfg.data.path is None:
            raise click.ClickException("Parquet data source requires a path")
        frame = pd.read_parquet(cfg.data.path)
        price_column = cfg.data.parameters.get("price_column", "close")
        if price_column not in frame.columns:
            raise click.ClickException(f"Column '{price_column}' not present in {cfg.data.path}")
        series = frame[price_column].astype(float).to_numpy()
        symbol = cfg.data.parameters.get("symbol", cfg.metadata.get("symbol", "asset"))
        return series, symbol
    raise click.ClickException(f"Unsupported data source kind: {cfg.data.kind}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to backtest YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default backtest config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.option("--validate-only", is_flag=True, help="Validate configuration without executing the backtest.")
@click.pass_context
def backtest(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    output: Path | None,
    validate_only: bool,
) -> None:
    """Execute an event-driven backtest and store the performance report."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when using --generate-config")
        _handle_generate(manager, "backtest", output)
        return
    if config is None:
        raise click.UsageError("--config is required for backtesting")

    cfg = manager.load_config(config, BacktestConfig)
    if validate_only:
        click.echo(f"Configuration {config} is valid.")
        return

    prices, symbol = _load_prices(cfg)
    signal_fn = _resolve_signal_callable(cfg.strategy)
    engine = EventDrivenBacktestEngine()

    latency_cfg = LatencyConfig(**{k: int(v) for k, v in cfg.execution.latency.items()})
    order_book = OrderBookConfig()
    slippage = SlippageConfig()

    result = engine.run(
        prices,
        signal_fn,
        fee=cfg.execution.fee_bps * 1e-4,
        initial_capital=cfg.execution.initial_capital,
        strategy_name=cfg.name,
        latency=latency_cfg,
        order_book=order_book,
        slippage=slippage,
        chunk_size=cfg.execution.chunk_size,
    )

    payload: Dict[str, Any] = {
        "name": cfg.name,
        "symbol": symbol,
        "pnl": result.pnl,
        "max_drawdown": result.max_dd,
        "trades": result.trades,
        "slippage_cost": result.slippage_cost,
        "latency_steps": result.latency_steps,
        "report_path": str(result.report_path) if result.report_path else None,
    }
    if result.performance is not None:
        payload["performance"] = result.performance.as_dict()
    if result.equity_curve is not None:
        payload["equity_curve"] = result.equity_curve.tolist()

    manager.write_json(cfg.results_path, payload)
    catalog = FeatureCatalog(cfg.catalog.path)
    lineage: List[str] = []
    if cfg.data.path is not None:
        lineage.append(str(cfg.data.path))
    catalog.register(cfg.name, cfg.results_path, config=cfg, lineage=lineage, metadata=cfg.metadata)
    versioner = DataVersionManager(cfg.versioning)
    versioner.snapshot(cfg.results_path, push=bool(cfg.versioning.remote))
    click.echo(f"Backtest results written to {cfg.results_path}")


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to optimisation YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default optimisation config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.option("--validate-only", is_flag=True, help="Validate configuration without running optimisation.")
@click.pass_context
def optimize(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    output: Path | None,
    validate_only: bool,
) -> None:
    """Run a simple grid-search optimiser using the configured objective."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when using --generate-config")
        _handle_generate(manager, "optimize", output)
        return
    if config is None:
        raise click.UsageError("--config is required for optimisation")

    cfg = manager.load_config(config, OptimizeConfig)
    if validate_only:
        click.echo(f"Configuration {config} is valid.")
        return

    objective = _load_callable(cfg.objective)
    keys = list(cfg.search_space.keys())
    values = [list(cfg.search_space[key]) for key in keys]
    direction = 1 if cfg.direction == "maximize" else -1
    evaluations: List[Dict[str, Any]] = []
    best_score = float("-inf") if cfg.direction == "maximize" else float("inf")
    best_params: Dict[str, Any] | None = None

    for trial_index, candidate in enumerate(itertools.product(*values), start=1):
        if cfg.max_trials is not None and trial_index > cfg.max_trials:
            break
        params = dict(zip(keys, candidate, strict=True))
        try:
            score = objective(params, **cfg.context)
        except Exception as exc:  # pragma: no cover - propagate failure details
            raise click.ClickException(f"Objective evaluation failed for params {params}: {exc}") from exc
        evaluations.append({"params": params, "score": score})
        adjusted = score * direction
        current_best = best_score * direction
        if best_params is None or adjusted > current_best:
            best_score = score
            best_params = params

    payload = {
        "name": cfg.name,
        "best_params": best_params,
        "best_score": best_score,
        "direction": cfg.direction,
        "evaluations": evaluations,
    }
    manager.write_json(cfg.results_path, payload)
    catalog = FeatureCatalog(cfg.catalog.path)
    catalog.register(cfg.name, cfg.results_path, config=cfg, lineage=[], metadata=cfg.metadata)
    versioner = DataVersionManager(cfg.versioning)
    versioner.snapshot(cfg.results_path, push=bool(cfg.versioning.remote))
    click.echo(f"Optimisation results written to {cfg.results_path}")


@cli.command(name="exec")
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to execution YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default execution config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.option("--validate-only", is_flag=True, help="Validate configuration without producing a plan.")
@click.pass_context
def exec_command(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    output: Path | None,
    validate_only: bool,
) -> None:
    """Create an execution plan artefact ready for deployment."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when using --generate-config")
        _handle_generate(manager, "exec", output)
        return
    if config is None:
        raise click.UsageError("--config is required for execution planning")

    cfg = manager.load_config(config, ExecConfig)
    if validate_only:
        click.echo(f"Configuration {config} is valid.")
        return

    signal_fn = _resolve_signal_callable(cfg.strategy)
    preview_prices = np.asarray(cfg.metadata.get("preview_prices", np.linspace(100.0, 101.0, 32)), dtype=float)
    preview_signal = signal_fn(preview_prices)
    plan = {
        "name": cfg.name,
        "broker": cfg.broker,
        "risk": cfg.risk,
        "signal_preview": preview_signal.tolist(),
        "preview_prices": preview_prices.tolist(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    plan_path = Path(cfg.metadata.get("plan_path", f"reports/exec/{cfg.name}.json")).resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    catalog = FeatureCatalog(cfg.catalog.path)
    catalog.register(cfg.name, plan_path, config=cfg, lineage=[], metadata=cfg.metadata)
    versioner = DataVersionManager(cfg.versioning)
    versioner.snapshot(plan_path, push=bool(cfg.versioning.remote))
    click.echo(f"Execution plan written to {plan_path}")


def _load_artifact_metadata(catalog: FeatureCatalog, path: Path) -> Dict[str, Any]:
    entry = catalog.find(path.stem)
    checksum = FeatureCatalog._checksum(path)
    metadata = {
        "name": path.stem,
        "path": str(path.resolve()),
        "checksum": checksum,
        "config_hash": entry.config_hash if entry else "",
        "timestamp": entry.timestamp if entry else "",
        "metadata": entry.metadata if entry else {},
    }
    return metadata


@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Path to report YAML config.")
@click.option("--generate-config", is_flag=True, help="Write the default report config template.")
@click.option("--output", type=click.Path(path_type=Path), help="Destination for generated template.")
@click.option("--validate-only", is_flag=True, help="Validate configuration without rendering the report.")
@click.pass_context
def report(
    ctx: click.Context,
    config: Path | None,
    generate_config: bool,
    output: Path | None,
    validate_only: bool,
) -> None:
    """Render a consolidated analytics report."""

    manager = _get_manager(ctx)
    if generate_config:
        if output is None:
            raise click.UsageError("--output must be provided when using --generate-config")
        _handle_generate(manager, "report", output)
        return
    if config is None:
        raise click.UsageError("--config is required for report generation")

    cfg = manager.load_config(config, ReportConfig)
    if validate_only:
        click.echo(f"Configuration {config} is valid.")
        return

    catalog = FeatureCatalog(cfg.catalog.path)
    artifacts: List[Dict[str, Any]] = []
    for input_path in cfg.inputs:
        if not input_path.exists():
            raise click.ClickException(f"Input artifact {input_path} does not exist")
        artifacts.append(_load_artifact_metadata(catalog, input_path))

    template_path = cfg.template or DEFAULT_TEMPLATES_DIR / "report.md.j2"
    template_manager = ConfigTemplateManager(template_path.parent)
    template = template_manager.environment.get_template(template_path.name)
    rendered = template.render(
        config=cfg,
        artifacts=artifacts,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(rendered, encoding="utf-8")

    catalog.register(cfg.name, cfg.output_path, config=cfg, lineage=[str(p) for p in cfg.inputs], metadata=cfg.metadata)
    versioner = DataVersionManager(cfg.versioning)
    versioner.snapshot(cfg.output_path, push=bool(cfg.versioning.remote))
    click.echo(f"Report written to {cfg.output_path}")


if __name__ == "__main__":  # pragma: no cover
    cli()
