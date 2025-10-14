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
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import walk_forward
from core.data.ingestion import DataIngestor
from core.indicators.entropy import delta_entropy, entropy
from core.indicators.hurst import hurst_exponent
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci
from core.phase.detector import composite_transition, phase_flags
from observability.tracing import activate_traceparent, current_traceparent, get_tracer

LOGGER = logging.getLogger(__name__)


def signal_from_indicators(prices: np.ndarray, window: int = 200) -> np.ndarray:
    """Derive a regime-aware trading signal from synchrony and entropy features.

    Args:
        prices: One-dimensional price series.
        window: Lookback window for phase, entropy, and curvature indicators.

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
    for t in range(window, n):
        p = prices[:t+1]
        phases = compute_phase(p)
        R = kuramoto_order(phases[-window:])
        H = entropy(p[-window:])
        dH = delta_entropy(p, window=window)
        G = build_price_graph(p[-window:], delta=0.005)
        kappa = mean_ricci(G)
        # basic decision
        comp = composite_transition(R, dH, kappa, H)
        # map comp to {-1,0,1}
        if comp > 0.15 and dH < 0 and kappa < 0:
            sig[t] = 1
        elif comp < -0.15 and dH > 0:
            sig[t] = -1
        else:
            sig[t] = sig[t-1]
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


def _enrich_with_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the active traceparent to ``payload`` when available."""
    traceparent = current_traceparent()
    if not traceparent:
        return payload
    enriched = dict(payload)
    enriched["traceparent"] = traceparent
    return enriched


def compute_indicator_metrics(
    prices: np.ndarray,
    *,
    window: int = 200,
    bins: int = 64,
    delta: float = 0.005,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Return the indicator diagnostics shared by the CLI and dashboard."""

    series = np.asarray(prices, dtype=float)
    if series.ndim != 1:
        raise ValueError("Indicator diagnostics expect a one-dimensional price series")
    if series.size < window:
        raise ValueError("Price series must be at least as long as the lookback window")

    gpu_used = False
    phases: np.ndarray
    if use_gpu:
        try:
            from core.indicators.kuramoto import compute_phase_gpu  # type: ignore

            phases = compute_phase_gpu(series)
            gpu_used = True
        except Exception:  # pragma: no cover - optional dependency path
            LOGGER.warning(
                "GPU phase computation requested but CuPy backend unavailable; "
                "falling back to CPU"
            )
            phases = compute_phase(series)
    else:
        phases = compute_phase(series)

    lookback_prices = series[-window:]
    lookback_phases = phases[-window:]

    R = float(kuramoto_order(lookback_phases))
    H = float(entropy(lookback_prices, bins=bins))
    dH = float(delta_entropy(series, window=window, bins_range=(10, 50)))
    kappa_graph = build_price_graph(lookback_prices, delta=delta)
    kappa = float(mean_ricci(kappa_graph))
    Hs = float(hurst_exponent(lookback_prices))
    phase = phase_flags(R, dH, kappa, H)

    return {
        "R": R,
        "H": H,
        "delta_H": dH,
        "kappa_mean": kappa,
        "Hurst": Hs,
        "phase": phase,
        "gpu_used": gpu_used,
        "window": window,
        "bins": bins,
        "delta": delta,
    }


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

    df = pd.read_csv(args.csv)
    prices = df[args.price_col].to_numpy()
    metrics = compute_indicator_metrics(
        prices,
        window=args.window,
        bins=args.bins,
        delta=args.delta,
        use_gpu=getattr(args, "gpu", False),
    )
    print(
        json.dumps(
            _enrich_with_trace(metrics),
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
    df = pd.read_csv(args.csv)
    prices = df[args.price_col].to_numpy()
    sig = signal_from_indicators(prices, window=args.window)
    res = walk_forward(prices, lambda _: sig, fee=args.fee)
    out = {"pnl": res.pnl, "max_dd": res.max_dd, "trades": res.trades}
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
