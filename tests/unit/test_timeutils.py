from datetime import datetime, timezone

import pytest

from core.data.timeutils import (
    MarketCalendar,
    MarketCalendarRegistry,
    convert_timestamp,
    is_market_open,
    normalize_timestamp,
)


def test_normalize_timestamp_from_float() -> None:
    ts = normalize_timestamp(1_700_000_000.0)
    assert ts.tzinfo == timezone.utc
    assert ts.timestamp() == pytest.approx(1_700_000_000.0)


def test_normalize_timestamp_with_market() -> None:
    ts = normalize_timestamp(1_700_000_000.0, market="BINANCE")
    assert ts.tzinfo == timezone.utc


def test_convert_timestamp_changes_timezone() -> None:
    utc_dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ny_dt = convert_timestamp(utc_dt, "NYSE")
    assert ny_dt.tzinfo is not None
    assert ny_dt.tzinfo.utcoffset(ny_dt).total_seconds() == pytest.approx(-5 * 3600)


def test_is_market_open_handles_weekends() -> None:
    weekend = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)  # Saturday
    assert not is_market_open(weekend, "NYSE")


def test_registry_allows_custom_calendar() -> None:
    registry = MarketCalendarRegistry()
    custom = MarketCalendar(market="TEST", timezone="UTC")
    registry.register(custom)
    assert registry.get("TEST") == custom
