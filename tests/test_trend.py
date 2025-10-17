from datetime import datetime, timezone

from strategies.trend import TrendStrategy, TrendStrategyConfig


def _bar(price: float, ts: int) -> dict:
    return {
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 1000,
    }


def test_trend_strategy_generates_directional_signal():
    cfg = TrendStrategyConfig(ma_fast=2, ma_slow=4, atr_window=2)
    strat = TrendStrategy(cfg)
    bars = [_bar(100 + i, i) for i in range(6)]
    signal = None
    for bar in bars:
        signal = strat.generate_signals(bar)
    assert signal is not None
    assert signal.side in {"buy", "sell"}
    assert -1.0 <= signal.strength <= 1.0

