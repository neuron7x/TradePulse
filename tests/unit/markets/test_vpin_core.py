from __future__ import annotations

from datetime import datetime, timedelta, timezone

from markets.vpin.src.core.main import TradeTick, VPINCalculator, compute_vpin_series


def test_vpin_calculator_basic() -> None:
    calculator = VPINCalculator(bucket_volume=10, window=2, staleness=timedelta(minutes=1))
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ticks = [
        TradeTick(timestamp=base_time, price=100.0, volume=6, side="buy"),
        TradeTick(timestamp=base_time + timedelta(seconds=10), price=100.5, volume=6, side="sell"),
        TradeTick(timestamp=base_time + timedelta(minutes=2), price=101.0, volume=10, side="buy"),
        TradeTick(timestamp=base_time + timedelta(minutes=2, seconds=30), price=101.2, volume=10, side="buy"),
    ]
    results = []
    for tick in ticks:
        result = calculator.add_trade(tick)
        if result:
            results.append(result.value)
    assert results[-1] <= 1.0

    final = calculator.flush()
    assert final is not None
    assert final.value <= 1.0


def test_compute_vpin_series() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ticks = [
        TradeTick(timestamp=base_time + timedelta(seconds=i * 10), price=100 + i, volume=5, side="buy" if i % 2 == 0 else "sell")
        for i in range(10)
    ]
    results = compute_vpin_series(ticks, bucket_volume=5, window=3, staleness=timedelta(minutes=1))
    assert results
    assert all(0.0 <= result.value <= 1.0 for result in results)
