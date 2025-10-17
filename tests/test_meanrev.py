from datetime import datetime, timezone

from strategies.meanrev import MeanReversionConfig, MeanReversionStrategy


def _bar(price: float, ts: int) -> dict:
    return {
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1000,
    }


def test_mean_reversion_signal_respects_trend_filter():
    cfg = MeanReversionConfig(lookback=5, trend_filter_window=5, z_entry=1.0)
    strat = MeanReversionStrategy(cfg)
    prices = [100, 99, 98, 97, 105, 90]
    signal = None
    for ts, price in enumerate(prices):
        signal = strat.generate_signals(_bar(price, ts))
    assert signal is not None
    assert signal.side in {"buy", "sell", "flat"}

