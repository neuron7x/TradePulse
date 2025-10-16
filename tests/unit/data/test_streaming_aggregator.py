from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from core.data.models import InstrumentType, PriceTick
from src.data import DataIngestionCacheService, TickStreamAggregator


BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _make_tick(
    minutes: int,
    price: float | int | str,
    *,
    symbol: str = "BTC/USDT",
    venue: str = "BINANCE",
    volume: float | int | str = 1,
    instrument_type: InstrumentType = InstrumentType.SPOT,
) -> PriceTick:
    return PriceTick.create(
        symbol=symbol,
        venue=venue,
        price=price,
        volume=volume,
        timestamp=BASE_TS + timedelta(minutes=minutes),
        instrument_type=instrument_type,
    )


def test_tick_stream_aggregator_merges_sources_and_detects_gaps() -> None:
    cache_service = DataIngestionCacheService()
    aggregator = TickStreamAggregator(cache_service=cache_service, timeframe="1min")

    historical = [_make_tick(0, "30000"), _make_tick(1, "30010")]
    live = [_make_tick(1, "30020"), _make_tick(2, "30030")]

    result = aggregator.synchronise(
        symbol="BTC/USDT",
        venue="BINANCE",
        instrument_type=InstrumentType.SPOT,
        historical=historical,
        live=live,
        start=BASE_TS,
        end=BASE_TS + timedelta(minutes=3),
    )

    frame = result.frame
    assert frame.index.tz == UTC
    assert frame.index.is_monotonic_increasing
    assert list(frame.index) == [
        pd.Timestamp(BASE_TS),
        pd.Timestamp(BASE_TS + timedelta(minutes=1)),
        pd.Timestamp(BASE_TS + timedelta(minutes=2)),
    ]
    assert frame.loc[pd.Timestamp(BASE_TS + timedelta(minutes=1))]["price"] == pytest.approx(30020.0)
    assert frame.loc[pd.Timestamp(BASE_TS + timedelta(minutes=2))]["price"] == pytest.approx(30030.0)

    assert len(result.backfill_plan.gaps) == 1
    gap = result.backfill_plan.gaps[0]
    assert gap.start == pd.Timestamp(BASE_TS + timedelta(minutes=3))
    assert gap.end == pd.Timestamp(BASE_TS + timedelta(minutes=4))


def test_tick_stream_aggregator_backfills_gaps_via_callback() -> None:
    cache_service = DataIngestionCacheService()
    aggregator = TickStreamAggregator(cache_service=cache_service, timeframe="1min")

    historical = [_make_tick(0, "30000"), _make_tick(2, "30040")]
    fetched: list[tuple[datetime, datetime]] = []

    def fetcher(start: datetime, end: datetime) -> list[PriceTick]:
        fetched.append((start, end))
        assert start.tzinfo is not None and end.tzinfo is not None
        return [_make_tick(1, "30020")]

    result = aggregator.synchronise(
        symbol="BTC/USDT",
        venue="BINANCE",
        instrument_type=InstrumentType.SPOT,
        historical=historical,
        start=BASE_TS,
        end=BASE_TS + timedelta(minutes=2),
        gap_fetcher=fetcher,
    )

    assert fetched == [
        (
            pd.Timestamp(BASE_TS + timedelta(minutes=1)).to_pydatetime(),
            pd.Timestamp(BASE_TS + timedelta(minutes=2)).to_pydatetime(),
        )
    ]
    assert not result.backfill_plan.gaps

    frame = result.frame
    assert frame.index.tz == UTC
    assert len(frame) == 3
    assert frame.loc[pd.Timestamp(BASE_TS + timedelta(minutes=1))]["price"] == pytest.approx(30020.0)


def test_tick_stream_aggregator_rejects_mismatched_metadata() -> None:
    cache_service = DataIngestionCacheService()
    aggregator = TickStreamAggregator(cache_service=cache_service, timeframe="1min")

    historical = [_make_tick(0, "30000"), _make_tick(1, "1500", symbol="ETH/USDT")]

    with pytest.raises(ValueError, match="Tick symbol does not match aggregation key"):
        aggregator.synchronise(
            symbol="BTC/USDT",
            venue="BINANCE",
            instrument_type=InstrumentType.SPOT,
            historical=historical,
            start=BASE_TS,
            end=BASE_TS + timedelta(minutes=1),
        )
