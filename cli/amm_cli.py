from __future__ import annotations
import asyncio, csv, argparse
from prometheus_client import start_http_server
from core.neuro.amm import AdaptiveMarketMind, AMMConfig
from analytics.amm_metrics import publish_metrics, timed_update

async def stream_csv(path: str):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row.get("x", 0.0))
            R = float(row.get("R", 0.5))
            kappa = float(row.get("kappa", 0.0))
            H = float(row.get("H", 0.0)) if "H" in row and row["H"] != "" else None
            yield x, R, kappa, H

async def run(path: str, symbol: str, tf: str, metrics_port: int):
    start_http_server(metrics_port)
    amm = AdaptiveMarketMind(AMMConfig())
    async for x,R,kappa,H in stream_csv(path):
        with timed_update(symbol, tf):
            out = await amm.aupdate(x,R,kappa,H)
        publish_metrics(symbol, tf, out, k=amm.gain, theta=amm.threshold, q_hi=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: x,R,kappa[,H]")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1m")
    ap.add_argument("--metrics-port", type=int, default=9095)
    args = ap.parse_args()
    asyncio.run(run(args.csv, args.symbol, args.tf, args.metrics_port))

if __name__ == "__main__":
    main()
