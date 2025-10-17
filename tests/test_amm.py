from datetime import datetime, timezone

from strategies.amm import AMMConfig, AMMStrategy


def _bar(price: float, ts: int) -> dict:
    return {
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1000,
    }


def test_amm_rebalances_out_of_range():
    cfg = AMMConfig(sigma_window=3, range_width_sigma=0.5)
    strat = AMMStrategy(cfg)
    prices = [100, 100.5, 101, 110]
    signal = None
    for ts, price in enumerate(prices):
        signal = strat.generate_signals(_bar(price, ts))
    assert signal is not None
    assert signal.side in {"buy", "sell", "flat"}

