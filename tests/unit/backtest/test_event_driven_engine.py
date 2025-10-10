"""Tests for the event-driven backtest engine."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import LatencyConfig, WalkForwardEngine
from backtest.event_driven import (
    CSVChunkDataHandler,
    EventDrivenBacktestEngine,
)


def _signal_function(prices: np.ndarray) -> np.ndarray:
    if prices.size == 0:
        return np.array([], dtype=float)
    diffs = np.diff(prices, prepend=prices[0])
    signals = np.where(diffs > 0, 1.0, -1.0)
    signals[0] = 0.0
    return signals


def test_event_engine_matches_walk_forward() -> None:
    prices = np.array([100.0, 101.0, 100.5, 102.0, 101.5, 103.0])
    vector_engine = WalkForwardEngine()
    expected = vector_engine.run(
        prices,
        _signal_function,
        fee=0.0,
        initial_capital=1_000.0,
    )

    engine = EventDrivenBacktestEngine()
    result = engine.run(
        prices,
        _signal_function,
        fee=0.0,
        initial_capital=1_000.0,
        chunk_size=2,
    )

    overlap = min(result.equity_curve.size, expected.equity_curve.size)
    assert np.allclose(result.equity_curve[-overlap:], expected.equity_curve[-overlap:])
    assert np.isclose(result.equity_curve[-1], expected.equity_curve[-1])
    assert result.trades == expected.trades
    assert result.latency_steps == expected.latency_steps
    assert np.isclose(result.pnl, expected.pnl)


def test_event_engine_with_latency_matches_walk_forward() -> None:
    prices = np.linspace(100, 110, num=8)
    latency = LatencyConfig(signal_to_order=1, order_to_execution=1, execution_to_fill=1)

    vector_engine = WalkForwardEngine()
    expected = vector_engine.run(
        prices,
        _signal_function,
        fee=0.0,
        initial_capital=500.0,
        latency=latency,
    )

    engine = EventDrivenBacktestEngine()
    result = engine.run(
        prices,
        _signal_function,
        fee=0.0,
        initial_capital=500.0,
        latency=latency,
        chunk_size=3,
    )

    overlap = min(result.equity_curve.size, expected.equity_curve.size)
    assert np.allclose(result.equity_curve[-overlap:], expected.equity_curve[-overlap:])
    assert np.isclose(result.equity_curve[-1], expected.equity_curve[-1])
    assert result.trades == expected.trades
    assert result.latency_steps == expected.latency_steps == latency.total_delay
    assert np.isclose(result.pnl, expected.pnl)


def test_csv_chunk_data_handler(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(datetime(2023, 1, 1), periods=5, freq="1min"),
            "close": np.linspace(100, 104, num=5),
        }
    )
    csv_path = tmp_path / "prices.csv"
    frame.to_csv(csv_path, index=False)

    handler = CSVChunkDataHandler(
        path=str(csv_path),
        price_column="close",
        symbol="TEST",
        chunk_size=2,
        parse_dates=True,
        date_column="timestamp",
    )

    steps: list[int] = []
    timestamps: list[datetime] = []
    total_events = 0
    for chunk in handler.stream():
        total_events += len(chunk)
        steps.extend(event.step for event in chunk)
        timestamps.extend(event.timestamp for event in chunk if event.timestamp is not None)
        assert all(event.symbol == "TEST" for event in chunk)

    assert total_events == len(frame)
    assert steps == list(range(len(frame)))
    assert len(timestamps) == len(frame)
