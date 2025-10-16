from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from application.system import (
    ExchangeAdapterConfig,
    LiveLoopSettings,
    TradePulseSystem,
    TradePulseSystemConfig,
)
from domain import OrderStatus, Signal, SignalAction
from execution.connectors import BinanceConnector


def _build_system(tmp_path: Path) -> TradePulseSystem:
    venue = ExchangeAdapterConfig(name="binance", connector=BinanceConnector())
    settings = LiveLoopSettings(state_dir=tmp_path / "state")
    config = TradePulseSystemConfig(venues=[venue], live_settings=settings)
    return TradePulseSystem(config)


def _data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sample.csv"


def test_tradepulse_system_generates_features_and_orders(tmp_path: Path) -> None:
    system = _build_system(tmp_path)

    market = system.ingest_csv(_data_path(), symbol="BTCUSDT", venue="BINANCE")
    assert not market.empty
    assert market.index.tz is not None
    assert system.last_ingestion_completed_at is not None
    assert system.last_ingestion_duration_seconds is not None
    assert system.last_ingestion_error is None
    assert system.last_ingestion_symbol == "BTCUSDT"

    features = system.build_feature_frame(market)
    assert "rsi" in features.columns

    def strategy(prices: np.ndarray) -> np.ndarray:
        threshold = float(prices.mean())
        return np.where(prices > threshold, 1.0, -1.0)

    signals = system.generate_signals(features, strategy=strategy)
    assert signals
    assert all(signal.symbol == "BTCUSDT" for signal in signals)
    assert system.last_signal_generated_at is not None
    assert system.last_signal_latency_seconds is not None
    assert system.last_signal_error is None

    payloads = system.signals_to_dtos(signals)
    assert payloads[-1]["symbol"] == "BTCUSDT"

    loop = system.ensure_live_loop()
    assert loop is system.ensure_live_loop()  # idempotent

    terminal_signal = signals[-1]
    order = system.submit_signal(
        terminal_signal,
        venue="binance",
        quantity=0.25,
        price=float(features[system.feature_pipeline.config.price_col].iloc[-1]),
    )

    assert order.symbol == "BTCUSDT"
    assert order.status == OrderStatus.PENDING
    assert order.side.value in {"buy", "sell"}
    assert system.last_execution_submission_at is not None
    assert system.last_execution_error is None


def test_tradepulse_system_rejects_hold_signal(tmp_path: Path) -> None:
    system = _build_system(tmp_path)
    signal = Signal(symbol="BTCUSDT", action=SignalAction.HOLD, confidence=0.2)

    with pytest.raises(ValueError):
        system.submit_signal(signal, venue="binance", quantity=1.0)


def test_generate_signals_filters_invalid_scores(tmp_path: Path) -> None:
    system = _build_system(tmp_path)

    index = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    feature_frame = pd.DataFrame(
        {
            "close": [100.0, 100.5, 101.0, 101.5],
            "feature": [0.1, 0.2, 0.3, 0.4],
        },
        index=index,
    )

    def strategy(_prices: np.ndarray) -> np.ndarray:
        return np.array([0.5, np.nan, np.inf, -0.75])

    signals = system.generate_signals(feature_frame, strategy=strategy, symbol="BTCUSDT")

    assert len(signals) == 2
    assert {signal.action for signal in signals} == {SignalAction.BUY, SignalAction.SELL}
    assert all(np.isfinite(signal.metadata["score"]) for signal in signals)
