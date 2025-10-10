#!/usr/bin/env python3
"""Nightly smoke end-to-end pipeline for TradePulse."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import walk_forward, Result  # noqa: E402
from core.data.ingestion import DataIngestor, Ticker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TradePulse smoke E2E pipeline.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "sample.csv", help="Path to CSV source data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "smoke-e2e",
        help="Directory for smoke pipeline artifacts.",
    )
    parser.add_argument("--seed", type=int, default=20240615, help="Seed used for deterministic outputs.")
    parser.add_argument(
        "--fee",
        type=float,
        default=0.0005,
        help="Trading fee applied during the backtest stage.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=12,
        help="Lookback window for deterministic momentum signal construction.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def run_cli_analyze(csv_path: Path, seed: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", str(seed))
    env["TRADEPULSE_SMOKE_SEED"] = str(seed)
    cmd = [sys.executable, "-m", "interfaces.cli", "analyze", "--csv", str(csv_path)]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    output = result.stdout.strip()
    if not output:
        raise RuntimeError("CLI analyze produced no output")
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse CLI analyze JSON: {output}") from exc
    return payload


def ingest_prices(csv_path: Path) -> list[Ticker]:
    ingestor = DataIngestor()
    ticks: list[Ticker] = []
    ingestor.historical_csv(str(csv_path), ticks.append, required_fields=("ts", "price", "volume"))
    if not ticks:
        raise RuntimeError("No ticks ingested from CSV")
    return ticks


def build_signal_function(
    metrics: Dict[str, Any],
    *,
    window: int,
) -> Callable[[np.ndarray], np.ndarray]:
    delta_entropy = float(metrics.get("delta_H", 0.0))
    kuramoto_bias = float(metrics.get("kappa_mean", 0.0))

    def _signal(prices: np.ndarray) -> np.ndarray:
        if prices.size == 0:
            return np.array([], dtype=float)
        shifted = np.roll(prices, 1)
        shifted[0] = prices[0]
        momentum = prices - shifted
        # rolling mean momentum for smoothing
        if window > 1 and momentum.size >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(momentum, kernel, mode="same")
        else:
            smoothed = momentum
        scale = float(np.std(smoothed)) or 1.0
        normalized = smoothed / scale
        bias = 0.25 if delta_entropy < 0 else -0.25
        curvature = -0.1 if kuramoto_bias < 0 else 0.1
        combined = normalized + bias + curvature
        signals = np.where(combined > 0.15, 1.0, np.where(combined < -0.15, -1.0, 0.0))
        signals = signals.astype(float)
        signals[0] = 0.0
        return signals

    return _signal


def run_backtest(prices: np.ndarray, signal_fn: Callable[[np.ndarray], np.ndarray], fee: float) -> Result:
    return walk_forward(prices, signal_fn, fee=fee, strategy_name="smoke_e2e")


def summarise_result(result: Result, ticks: list[Ticker], metrics: Dict[str, Any]) -> Dict[str, Any]:
    report_path = result.report_path
    report_payload: Dict[str, Any] | None = None
    if report_path and report_path.exists():
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "ingested_ticks": len(ticks),
        "cli_metrics": metrics,
        "backtest": {
            "pnl": result.pnl,
            "max_drawdown": result.max_dd,
            "trades": result.trades,
            "report_path": str(report_path) if report_path else None,
            "report": report_payload,
        },
    }


def write_artifacts(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV source not found: {csv_path}")

    seed_everything(args.seed)

    metrics = run_cli_analyze(csv_path, args.seed)
    ticks = ingest_prices(csv_path)
    prices = np.array([float(t.price) for t in ticks], dtype=float)
    signal_fn = build_signal_function(metrics, window=args.momentum_window)
    result = run_backtest(prices, signal_fn, fee=args.fee)

    summary = summarise_result(result, ticks, metrics)
    summary["seed"] = args.seed

    write_artifacts(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
