# SPDX-License-Identifier: MIT
"""Regression tests for CLI ingestion using custom price columns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from interfaces import cli


def _write_price_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts", "close", "volume"])
        writer.writeheader()
        for idx in range(32):
            writer.writerow({
                "ts": str(float(idx)),
                "close": f"{100.0 + idx * 0.25}",
                "volume": f"{1000.0 + idx}",
            })


def _analyze_args(csv_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        csv=str(csv_path),
        price_col="close",
        window=8,
        bins=20,
        delta=0.01,
        config=None,
        gpu=False,
        traceparent=None,
    )


def _backtest_args(csv_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        csv=str(csv_path),
        price_col="close",
        window=8,
        fee=0.0005,
        config=None,
        gpu=False,
        traceparent=None,
    )


def test_cmd_analyze_accepts_custom_price_column(tmp_path, capsys) -> None:
    csv_path = tmp_path / "custom_prices.csv"
    _write_price_csv(csv_path)

    args = _analyze_args(csv_path)
    cli.cmd_analyze(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) >= {"R", "H", "delta_H", "kappa_mean", "Hurst", "phase"}
    assert all(np.isfinite(payload[key]) for key in ("R", "H", "delta_H", "kappa_mean", "Hurst"))


def test_cmd_backtest_accepts_custom_price_column(tmp_path, capsys) -> None:
    csv_path = tmp_path / "custom_prices.csv"
    _write_price_csv(csv_path)

    args = _backtest_args(csv_path)
    cli.cmd_backtest(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload) == {"pnl", "max_dd", "trades"}
    assert all(isinstance(payload[key], (int, float)) for key in payload)
