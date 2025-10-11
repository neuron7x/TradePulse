# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from observability.tracing import activate_traceparent, current_traceparent, get_tracer

from backtest.engine import walk_forward
from core.indicators.entropy import delta_entropy, entropy
from core.indicators.hurst import hurst_exponent
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci
from core.phase.detector import composite_transition, phase_flags

def signal_from_indicators(prices: np.ndarray, window: int = 200) -> np.ndarray:
    """Return -1/0/1 based on composite transition signal and simple thresholds."""
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
    """Return the PyYAML module, raising a helpful error when missing."""

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "YAML configuration support requires the 'PyYAML' package. "
            "Install it via 'pip install PyYAML' or omit the --config option."
        ) from exc
    return yaml


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge configuration overrides from a YAML file into CLI arguments."""

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
    if isinstance(data_section, Mapping) and "path" in data_section and not getattr(args, "csv", None):
        args.csv = data_section["path"]

    return args

def _enrich_with_trace(payload: dict[str, Any]) -> dict[str, Any]:
    traceparent = current_traceparent()
    if not traceparent:
        return payload
    enriched = dict(payload)
    enriched["traceparent"] = traceparent
    return enriched


def cmd_analyze(args):
    args = _apply_config(args)

    df = pd.read_csv(args.csv)
    prices = df[args.price_col].to_numpy()
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
                }
            ),
            indent=2,
        )
    )

def cmd_backtest(args):
    args = _apply_config(args)
    df = pd.read_csv(args.csv)
    prices = df[args.price_col].to_numpy()
    sig = signal_from_indicators(prices, window=args.window)
    from backtest.engine import walk_forward
    res = walk_forward(prices, lambda _: sig, fee=args.fee)
    out = {"pnl": res.pnl, "max_dd": res.max_dd, "trades": res.trades}
    print(json.dumps(_enrich_with_trace(out), indent=2))

def cmd_live(args):
    args = _apply_config(args)
    from core.data.ingestion import DataIngestor, Ticker
    import queue, threading
    q = queue.Queue(maxsize=10000)
    def on_tick(t: Ticker):
        q.put(t)
    if args.source == "csv":
        DataIngestor().historical_csv(args.path, on_tick)
    else:
        raise SystemExit("Only CSV demo supported in CLI live mode.")

    window = args.window
    prices = []
    while not q.empty():
        t = q.get()
        prices.append(float(t.price))
        if len(prices) >= window:
            p = np.array(prices[-window:])
            phases = compute_phase(p)
            R = kuramoto_order(phases)
            H = entropy(p)
            dH = delta_entropy(np.array(prices), window=window)
            kappa = mean_ricci(build_price_graph(p, delta=0.005))
            print(
                json.dumps(
                    _enrich_with_trace(
                        {
                            "ts": t.timestamp.isoformat(),
                            "R": float(R),
                            "H": float(H),
                            "delta_H": float(dH),
                            "kappa_mean": float(kappa),
                        }
                    )
                )
            )


def _run_with_trace_context(cmd_name: str, args: argparse.Namespace) -> None:
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
    pl.add_argument("--source", choices=["csv"], default="csv")
    pl.add_argument("--path", required=True)
    pl.add_argument("--window", type=int, default=200)
    pl.add_argument("--config", help="YAML config path", default=None)
    pl.add_argument("--gpu", action="store_true")
    pl.add_argument("--traceparent", default=None, help=trace_arg_help)
    pl.set_defaults(func=cmd_live)

    args = p.parse_args()
    _run_with_trace_context(args.cmd, args)

if __name__ == "__main__":
    main()
