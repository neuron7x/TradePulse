# SPDX-License-Identifier: MIT

from __future__ import annotations
import json, argparse, numpy as np, pandas as pd, time
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.entropy import entropy, delta_entropy
from core.indicators.hurst import hurst_exponent
from core.indicators.ricci import build_price_graph, mean_ricci
from core.phase.detector import phase_flags, composite_transition
from backtest.engine import walk_forward

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


def _apply_config(args):
    if getattr(args, "config", None):
        import yaml, types
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
        # simple override: indicators.window, indicators.bins, execution.risk, etc.
        ind = cfg.get("indicators", {})
        for k in ["window","bins","delta"]:
            if k in ind:
                setattr(args, k if k!="delta" else "delta", ind[k])
        data = cfg.get("data", {})
        if "path" in data and not getattr(args, "csv", None):
            args.csv = data["path"]
    return args

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
    print(json.dumps({"R": float(R), "H": float(H), "delta_H": float(dH), "kappa_mean": float(kappa), "Hurst": float(Hs), "phase": phase}, indent=2))

def cmd_backtest(args):
    args = _apply_config(args)
    df = pd.read_csv(args.csv)
    prices = df[args.price_col].to_numpy()
    sig = signal_from_indicators(prices, window=args.window)
    from backtest.engine import walk_forward
    res = walk_forward(prices, lambda _: sig, fee=args.fee)
    out = {"pnl": res.pnl, "max_dd": res.max_dd, "trades": res.trades}
    print(json.dumps(out, indent=2))

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
        prices.append(t.price)
        if len(prices) >= window:
            p = np.array(prices[-window:])
            phases = compute_phase(p)
            R = kuramoto_order(phases)
            H = entropy(p)
            dH = delta_entropy(np.array(prices), window=window)
            kappa = mean_ricci(build_price_graph(p, delta=0.005))
            print(json.dumps({"ts": t.ts, "R": float(R), "H": float(H), "delta_H": float(dH), "kappa_mean": float(kappa)}))

def main():
    import yaml, os
    p = argparse.ArgumentParser(prog="tradepulse")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("analyze")
    pa.add_argument("--csv", required=True)
    pa.add_argument("--price-col", default="price")
    pa.add_argument("--window", type=int, default=200)
    pa.add_argument("--bins", type=int, default=30)
    pa.add_argument("--delta", type=float, default=0.005)
    pa.add_argument("--config", help="YAML config path", default=None)
    pa.add_argument("--gpu", action="store_true")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("backtest")
    pb.add_argument("--csv", required=True)
    pb.add_argument("--price-col", default="price")
    pb.add_argument("--window", type=int, default=200)
    pb.add_argument("--fee", type=float, default=0.0005)
    pb.add_argument("--config", help="YAML config path", default=None)
    pb.add_argument("--gpu", action="store_true")
    pb.set_defaults(func=cmd_backtest)

    pl = sub.add_parser("live")
    pl.add_argument("--source", choices=["csv"], default="csv")
    pl.add_argument("--path", required=True)
    pl.add_argument("--window", type=int, default=200)
    pl.add_argument("--config", help="YAML config path", default=None)
    pl.add_argument("--gpu", action="store_true")
    pl.set_defaults(func=cmd_live)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
