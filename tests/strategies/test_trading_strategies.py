"""Tests for algorithmic trading strategies."""

import numpy as np
import pandas as pd
import pytest

from core.strategies import HurstVPINStrategy, KuramotoStrategy


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample OHLCV data for testing."""

    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "close": rng.normal(100, 10, size=100),
            "volume": rng.integers(1_000, 10_000, size=100),
            "buy_volume": rng.integers(500, 5_000, size=100),
            "sell_volume": rng.integers(500, 5_000, size=100),
        },
        index=dates,
    )


def test_kuramoto_strategy(sample_data: pd.DataFrame) -> None:
    """Test KuramotoStrategy signal generation."""

    params = {"window": 50, "coupling": 1.0, "sync_threshold": 0.5}
    strategy = KuramotoStrategy(symbol="TEST", params=params)
    signals = strategy.generate_signals(sample_data)

    assert isinstance(signals, pd.DataFrame)
    assert set(signals.columns) == {"timestamp", "symbol", "signal", "confidence"}
    assert signals["symbol"].eq("TEST").all()
    assert signals["signal"].isin(["Buy", "Sell", "Hold"]).all()
    assert ((signals["confidence"] >= 0) & (signals["confidence"] <= 1)).all()


def test_hurst_vpin_strategy(sample_data: pd.DataFrame) -> None:
    """Test HurstVPINStrategy signal generation."""

    params = {
        "hurst_window": 50,
        "vpin_bucket_size": 20,
        "hurst_trend_threshold": 0.6,
        "vpin_safe_threshold": 0.5,
    }
    strategy = HurstVPINStrategy(symbol="TEST", params=params)
    signals = strategy.generate_signals(sample_data)

    assert isinstance(signals, pd.DataFrame)
    assert set(signals.columns) == {"timestamp", "symbol", "signal", "confidence"}
    assert signals["symbol"].eq("TEST").all()
    assert signals["signal"].isin(["Buy", "Sell", "Hold"]).all()
    assert ((signals["confidence"] >= 0) & (signals["confidence"] <= 1)).all()


def test_edge_cases_kuramoto() -> None:
    """Test KuramotoStrategy with edge cases (NaN, empty data)."""

    strategy = KuramotoStrategy(symbol="TEST", params={"window": 50, "sync_threshold": 0.5})
    empty = pd.DataFrame({"close": []})
    signals_empty = strategy.generate_signals(empty)
    assert signals_empty.empty

    nan_data = pd.DataFrame(
        {"close": [np.nan] * 100},
        index=pd.date_range("2025-01-01", periods=100, freq="h"),
    )
    signals_nan = strategy.generate_signals(nan_data)
    assert signals_nan["signal"].eq("Hold").all()
    assert ((signals_nan["confidence"] >= 0) & (signals_nan["confidence"] <= 1)).all()


def test_kuramoto_strategy_small_window_breakout() -> None:
    """KuramotoStrategy should produce actionable signals for small windows."""

    data = pd.DataFrame(
        {"close": [100.0, 101.0, 102.5, 104.0, 106.0, 108.5]},
        index=pd.date_range("2025-01-01", periods=6, freq="h"),
    )
    params = {"window": 3, "coupling": 1.0, "sync_threshold": 0.1}
    strategy = KuramotoStrategy(symbol="TEST", params=params)

    signals = strategy.generate_signals(data)

    assert (signals["signal"] != "Hold").any()
