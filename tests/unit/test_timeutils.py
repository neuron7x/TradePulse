from datetime import datetime, timezone

import pandas as pd
import pytest

from core.data.timeutils import (
    MarketCalendar,
    MarketCalendarRegistry,
    convert_timestamp,
    get_market_calendar,
    get_timezone,
    is_market_open,
    normalize_timestamp,
    validate_bar_alignment,
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


def test_convert_timestamp_handles_dst_shift() -> None:
    before = datetime(2024, 3, 8, 19, 0, tzinfo=timezone.utc)
    after = datetime(2024, 3, 11, 19, 0, tzinfo=timezone.utc)

    before_local = convert_timestamp(before, "NYSE")
    after_local = convert_timestamp(after, "NYSE")

    assert before_local.hour == 14  # UTC-5
    assert after_local.hour == 15  # UTC-4 after DST transition


def test_is_market_open_handles_weekends() -> None:
    weekend = datetime(2024, 1, 6, 12, 0, tzinfo=timezone.utc)  # Saturday
    assert not is_market_open(weekend, "NYSE")


def test_is_market_open_respects_holidays() -> None:
    independence_day = datetime(2024, 7, 4, 15, 0, tzinfo=timezone.utc)
    assert not is_market_open(independence_day, "NYSE")


def test_registry_allows_custom_calendar() -> None:
    registry = MarketCalendarRegistry()
    custom = MarketCalendar(market="TEST", timezone="UTC")
    registry.register(custom)
    assert registry.get("TEST") == custom


def test_default_registry_includes_nasdaq() -> None:
    calendar = get_market_calendar("NASDAQ")
    assert calendar.timezone == "America/New_York"


def test_exchange_aliases_are_supported() -> None:
    nyse = get_market_calendar("NYSE")
    assert nyse is get_market_calendar("XNYS")
    assert get_market_calendar("24/7").market == "BINANCE"


def test_validate_bar_alignment_accepts_dst_boundary_minutes() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2024-03-08 20:55:00+00:00",
            "2024-03-08 20:56:00+00:00",
            "2024-03-08 20:57:00+00:00",
            "2024-03-08 20:58:00+00:00",
            "2024-03-08 20:59:00+00:00",
            "2024-03-11 13:30:00+00:00",
            "2024-03-11 13:31:00+00:00",
        ]
    )

    validate_bar_alignment(timestamps, market="NYSE", frequency="1min")


def test_validate_bar_alignment_detects_gaps() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2024-03-11 13:30:00+00:00",
            "2024-03-11 13:31:00+00:00",
            "2024-03-11 13:33:00+00:00",  # Missing 13:32
        ]
    )

    with pytest.raises(ValueError):
        validate_bar_alignment(timestamps, market="NYSE", frequency="1min")


def test_validate_bar_alignment_handles_always_open_market() -> None:
    timestamps = pd.date_range(
        "2024-01-01 00:00:00+00:00",
        periods=6,
        freq="5min",
        tz="UTC",
    )

    validate_bar_alignment(timestamps, market="BINANCE", frequency="5min")


def test_get_timezone_round_trip() -> None:
    tz = get_timezone("America/New_York")
    assert tz.key == "America/New_York"

    with pytest.raises(ValueError):
        get_timezone("Mars/Phobos")
