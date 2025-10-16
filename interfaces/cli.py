# SPDX-License-Identifier: MIT
"""Command-line interface orchestrating TradePulse analytics workflows.

The CLI glues together ingestion, indicator computation, backtesting, and live
trading bootstrap flows. It is the operational entry point referenced in
``docs/quickstart.md`` and ``docs/runbook_live_trading.md`` and emits tracing
metadata according to ``docs/monitoring.md``. Each command surfaces the
governance requirements outlined in ``docs/documentation_governance.md`` by
exposing structured outputs and traceparent propagation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import walk_forward
from core.data.quality_control import QualityReport, validate_and_quarantine
from core.data.ingestion import DataIngestor
from core.data.validation import TimeSeriesValidationConfig, ValueColumnConfig
from core.indicators.entropy import delta_entropy, entropy
from core.indicators.hurst import hurst_exponent
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci
from core.phase.detector import composite_transition, phase_flags
from core.utils import get_metrics_collector
from observability.tracing import activate_traceparent, current_traceparent, get_tracer


def signal_from_indicators(
    prices: np.ndarray,
    window: int = 200,
    *,
    max_workers: int | None = None,
    ricci_delta: float = 0.005,
) -> np.ndarray:
    """Derive a regime-aware trading signal from synchrony and entropy features.

    Args:
        prices: One-dimensional price series.
        window: Lookback window for phase, entropy, and curvature indicators.
        max_workers: Optional thread pool size for indicator fan-out. ``None``
            defaults to three workers covering entropy, delta entropy, and Ricci
            curvature. Values ``<= 1`` disable parallelism.
        ricci_delta: Step size used when constructing the Ricci price graph.

    Returns:
        np.ndarray: Array of integer signals in ``{-1, 0, 1}`` aligned with
        ``prices``.

    Examples:
        >>> prices = np.linspace(100, 105, 256)
        >>> signals = signal_from_indicators(prices, window=32)
        >>> set(np.unique(signals)).issubset({-1, 0, 1})
        True

    Notes:
        This routine mirrors the composite signal recipe in ``docs/quickstart.md``
        and is meant for demonstrations rather than production alpha generation.
    """
    n = len(prices)
    sig = np.zeros(n, dtype=int)
    worker_count = max_workers if max_workers is not None else 3
    executor: ThreadPoolExecutor | None = None
    if worker_count is not None and worker_count > 1:
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="signal-indicators")

    def _compute_ricci(window_prices: np.ndarray) -> float:
        graph = build_price_graph(window_prices, delta=ricci_delta)
        return mean_ricci(graph)

    try:
        for t in range(window, n):
            prefix = prices[: t + 1]
            window_prices = prefix[-window:]

            phases = compute_phase(prefix)
            R = kuramoto_order(phases[-window:])

            if executor is None:
                H = entropy(window_prices)
                dH = delta_entropy(prefix, window=window)
                kappa = _compute_ricci(window_prices)
            else:
                futures = {
                    "entropy": executor.submit(entropy, window_prices),
                    "delta_entropy": executor.submit(delta_entropy, prefix, window=window),
                    "ricci": executor.submit(_compute_ricci, window_prices),
                }
                H = futures["entropy"].result()
                dH = futures["delta_entropy"].result()
                kappa = futures["ricci"].result()

            comp = composite_transition(R, dH, kappa, H)
            if comp > 0.15 and dH < 0 and kappa < 0:
                sig[t] = 1
            elif comp < -0.15 and dH > 0:
                sig[t] = -1
            else:
                sig[t] = sig[t - 1]
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return sig


def _load_yaml() -> Any:
    """Return the PyYAML module, raising a helpful error when missing.

    Returns:
        Any: The imported PyYAML module.

    Raises:
        RuntimeError: If PyYAML is not installed.
    """

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "YAML configuration support requires the 'PyYAML' package. "
            "Install it via 'pip install PyYAML' or omit the --config option."
        ) from exc
    return yaml


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge configuration overrides from a YAML file into CLI arguments.

    Args:
        args: Namespace produced by :mod:`argparse`.

    Returns:
        argparse.Namespace: Updated namespace reflecting YAML overrides.
    """

    config_path = getattr(args, "config", None)
    if not config_path:
        return args

    yaml = _load_yaml()
    path = Path(config_path)
    config: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, Mapping):
                config = dict(loaded)

    indicators = config.get("indicators", {})
    if isinstance(indicators, Mapping):
        for key in ("window", "bins", "delta"):
            if key in indicators:
                setattr(args, key, indicators[key])

    data_section = config.get("data", {})
    if (
        isinstance(data_section, Mapping)
        and "path" in data_section
        and not getattr(args, "csv", None)
    ):
        args.csv = data_section["path"]

    return args


def _make_data_ingestor(csv_path: str | None = None) -> DataIngestor:
    """Return a :class:`DataIngestor` constrained to directories derived from ``csv_path``.

    Args:
        csv_path: Optional CSV location used to infer allowed directories.

    Returns:
        DataIngestor: Configured ingestor with restricted roots per
        ``docs/documentation_governance.md``.
    """

    allowed = None
    if csv_path:
        allowed = [Path(csv_path).expanduser().resolve(strict=False).parent]
    return DataIngestor(allowed_roots=allowed)


def _derive_symbol(args: argparse.Namespace) -> str:
    candidate = getattr(args, "symbol", None)
    if candidate:
        return str(candidate).upper()
    csv_path = getattr(args, "csv", None)
    if not csv_path:
        return "UNKNOWN"
    stem = Path(csv_path).stem
    return stem.upper() or "UNKNOWN"


def _ticks_to_frame(ticks: list[Any]) -> pd.DataFrame:
    timestamps = pd.to_datetime([tick.timestamp for tick in ticks], utc=True)
    prices = [float(tick.price) for tick in ticks]
    volumes = [float(tick.volume) for tick in ticks]
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes,
    })
    return frame


def _prepare_quality_report(report: QualityReport, *, price_column: str) -> QualityReport:
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if prepared.empty:
            return prepared
        if "timestamp" in prepared.columns:
            prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True, errors="coerce")
            prepared = prepared.sort_values("timestamp").reset_index(drop=True)
        if price_column != "price" and "price" in prepared.columns:
            prepared = prepared.rename(columns={"price": price_column})
        return prepared

    return QualityReport(
        clean=_normalise(report.clean),
        quarantined=_normalise(report.quarantined),
        duplicates=_normalise(report.duplicates),
        spikes=_normalise(report.spikes),
    )


def _serialise_value(value: Any) -> Any:
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
        return ts.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return value


def _frame_preview(frame: pd.DataFrame, *, limit: int = 10) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    preview = frame.head(limit).copy()
    for column in preview.columns:
        series = preview[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            converted = series.dt.tz_convert("UTC")
            preview[column] = [value.isoformat() if not pd.isna(value) else None for value in converted]
        else:
            preview[column] = series.apply(_serialise_value)
    return preview.to_dict(orient="records")


def _quality_payload(report: QualityReport) -> dict[str, Any]:
    return {
        "clean_rows": int(report.clean.shape[0]),
        "quarantined_rows": int(report.quarantined.shape[0]),
        "duplicates_rows": int(report.duplicates.shape[0]),
        "spikes_rows": int(report.spikes.shape[0]),
        "quarantined_preview": _frame_preview(report.quarantined),
        "duplicates_preview": _frame_preview(report.duplicates),
        "spikes_preview": _frame_preview(report.spikes),
    }


def _ingest_and_validate(args: argparse.Namespace) -> tuple[pd.DataFrame, QualityReport, str]:
    ingestor = _make_data_ingestor(args.csv)
    collector = get_metrics_collector()
    symbol = _derive_symbol(args)
    ticks: list[Any] = []

    with collector.measure_data_ingestion("csv", symbol) as ctx:
        ingestor.historical_csv(
            args.csv,
            ticks.append,
            required_fields=("ts", "price"),
            symbol=symbol,
            venue="CLI",
        )
        count = len(ticks)
        collector.record_tick_processed("csv", symbol, count)
        ctx.setdefault("rows", count)

    if not ticks:
        raise ValueError(f"No ticks were ingested from {args.csv}")

    frame = _ticks_to_frame(ticks)
    frame_length = len(frame)
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[
            ValueColumnConfig(name="price", dtype="float64", nullable=False),
            ValueColumnConfig(name="volume", dtype="float64", nullable=True),
        ],
        allow_extra_columns=True,
    )
    window_size = max(1, min(args.window, frame_length or 1))
    report = validate_and_quarantine(
        frame,
        config,
        window=window_size,
        price_column="price",
    )
    prepared = _prepare_quality_report(report, price_column=args.price_col)
    if prepared.clean.empty:
        raise ValueError("All ingested rows were quarantined; no clean data available.")
    return prepared.clean, prepared, symbol


def _enrich_with_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the active traceparent to ``payload`` when available."""
    traceparent = current_traceparent()
    if not traceparent:
        return payload
    enriched = dict(payload)
    enriched["traceparent"] = traceparent
    return enriched


def cmd_analyze(args):
    """Compute indicator diagnostics for a CSV price series.

    Args:
        args: Parsed :class:`argparse.Namespace` with CLI options. Expected
            attributes include ``csv``, ``price_col``, ``window``, ``bins``,
            ``delta``, ``gpu``, and ``traceparent``.

    Returns:
        None: Writes a JSON payload to stdout.

    Notes:
        Outputs a JSON payload suitable for audit pipelines described in
        ``docs/documentation_governance.md``. GPU acceleration is attempted when
        the ``--gpu`` flag is set and the CuPy-backed implementation is available.
    """
    args = _apply_config(args)

    clean_frame, quality_report, _ = _ingest_and_validate(args)
    prices = clean_frame[args.price_col].to_numpy(dtype=float, copy=False)
    from core.indicators.kuramoto import compute_phase_gpu
    phases = compute_phase_gpu(prices) if getattr(args,'gpu',False) else compute_phase(prices)
    R = kuramoto_order(phases[-args.window:])
    H = entropy(prices[-args.window:], bins=args.bins)
    dH = delta_entropy(prices, window=args.window, bins_range=(10,50))
    kappa = mean_ricci(build_price_graph(prices[-args.window:], delta=args.delta))
    Hs = hurst_exponent(prices[-args.window:])
    phase = phase_flags(R, dH, kappa, H)
    print(
        json.dumps(
            _enrich_with_trace(
                {
                    "R": float(R),
                    "H": float(H),
                    "delta_H": float(dH),
                    "kappa_mean": float(kappa),
                    "Hurst": float(Hs),
                    "phase": phase,
                    "quality": _quality_payload(quality_report),
                }
            ),
            indent=2,
        )
    )

def cmd_backtest(args):
    """Run a walk-forward backtest using the indicator composite signal.

    Args:
        args: Parsed :class:`argparse.Namespace` with options ``csv``,
            ``price_col``, ``window``, ``fee``, ``config``, ``gpu``, and
            ``traceparent``.

    Returns:
        None: Emits JSON summary statistics to stdout.

    Notes:
        Returns backtest statistics in JSON form to comply with
        ``docs/performance.md`` reporting guidance. The walk-forward engine is the
        same implementation invoked by automation in ``docs/runbook_live_trading.md``.
    """
    args = _apply_config(args)
    clean_frame, quality_report, symbol = _ingest_and_validate(args)
    prices = clean_frame[args.price_col].to_numpy(dtype=float, copy=False)
    sig = signal_from_indicators(prices, window=args.window)
    collector = get_metrics_collector()
    with collector.measure_backtest("cli.signal") as backtest_ctx:
        res = walk_forward(prices, lambda _: sig, fee=args.fee)
        backtest_ctx.update({"pnl": res.pnl, "max_dd": res.max_dd, "trades": res.trades})
    out = {
        "pnl": res.pnl,
        "max_dd": res.max_dd,
        "trades": res.trades,
        "quality": _quality_payload(quality_report),
        "symbol": symbol,
    }
    print(json.dumps(_enrich_with_trace(out), indent=2))

def cmd_live(args):
    """Bootstrap the live trading runner with risk and tracing configuration.

    Args:
        args: Parsed :class:`argparse.Namespace` including ``config``, ``venue``,
            ``state_dir``, ``cold_start``, ``metrics_port``, and ``traceparent``
            attributes.

    Returns:
        None: Executes the live runner with side effects only.

    Notes:
        Delegates to :class:`interfaces.live_runner.LiveTradingRunner`, ensuring
        the kill-switch and telemetry expectations from ``docs/admin_remote_control.md``
        and ``docs/monitoring.md`` are met. Raises :class:`FileNotFoundError` if
        the configuration file is missing.
    """
    from interfaces.live_runner import LiveTradingRunner

    config_path = Path(args.config).expanduser() if args.config else None
    venues = tuple(args.venue or ()) or None
    state_dir = Path(args.state_dir).expanduser() if args.state_dir else None
    metrics_port = args.metrics_port
    cold_start = bool(args.cold_start)

    runner = LiveTradingRunner(
        config_path,
        venues=venues,
        state_dir_override=state_dir,
        metrics_port=metrics_port,
    )
    runner.run(cold_start=cold_start)


def _run_with_trace_context(cmd_name: str, args: argparse.Namespace) -> None:
    """Execute a CLI command with traceparent propagation.

    Args:
        cmd_name: Name of the command for tracer labelling.
        args: Parsed arguments namespace.

    Returns:
        None.

    Notes:
        This helper ensures every CLI invocation participates in distributed
        tracing as outlined in ``docs/monitoring.md``.
    """
    tracer = get_tracer("tradepulse.cli")
    inbound = getattr(args, "traceparent", None) or os.environ.get("TRADEPULSE_TRACEPARENT")
    with activate_traceparent(inbound):
        with tracer.start_as_current_span(
            f"cli.{cmd_name}",
            attributes={"cli.command": cmd_name},
        ):
            outbound = current_traceparent()
            previous = os.environ.get("TRADEPULSE_TRACEPARENT")
            if outbound:
                os.environ["TRADEPULSE_TRACEPARENT"] = outbound
            try:
                args.func(args)
            finally:
                if outbound:
                    if previous is None:
                        os.environ.pop("TRADEPULSE_TRACEPARENT", None)
                    else:
                        os.environ["TRADEPULSE_TRACEPARENT"] = previous

def main():
    p = argparse.ArgumentParser(prog="tradepulse")
    sub = p.add_subparsers(dest="cmd", required=True)

    trace_arg_help = "W3C traceparent header used to join an existing trace"

    pa = sub.add_parser("analyze")
    pa.add_argument("--csv", required=True)
    pa.add_argument("--price-col", default="price")
    pa.add_argument("--window", type=int, default=200)
    pa.add_argument("--bins", type=int, default=30)
    pa.add_argument("--delta", type=float, default=0.005)
    pa.add_argument("--config", help="YAML config path", default=None)
    pa.add_argument("--gpu", action="store_true")
    pa.add_argument("--traceparent", default=None, help=trace_arg_help)
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("backtest")
    pb.add_argument("--csv", required=True)
    pb.add_argument("--price-col", default="price")
    pb.add_argument("--window", type=int, default=200)
    pb.add_argument("--fee", type=float, default=0.0005)
    pb.add_argument("--config", help="YAML config path", default=None)
    pb.add_argument("--gpu", action="store_true")
    pb.add_argument("--traceparent", default=None, help=trace_arg_help)
    pb.set_defaults(func=cmd_backtest)

    pl = sub.add_parser("live")
    pl.add_argument(
        "--config",
        default="configs/live/default.toml",
        help="Path to the TOML configuration describing venues and risk limits.",
    )
    pl.add_argument(
        "--venue",
        action="append",
        default=None,
        help="Restrict execution to specific venues (can be provided multiple times).",
    )
    pl.add_argument(
        "--state-dir",
        default=None,
        help="Override the state directory used for OMS state.",
    )
    pl.add_argument(
        "--cold-start",
        action="store_true",
        help="Skip reconciliation on startup.",
    )
    pl.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on a port.",
    )
    pl.add_argument("--traceparent", default=None, help=trace_arg_help)
    pl.set_defaults(func=cmd_live)

    args = p.parse_args()
    _run_with_trace_context(args.cmd, args)

if __name__ == "__main__":
    main()
