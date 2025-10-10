# SPDX-License-Identifier: MIT
"""Centralised timezone and market calendar helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Dict, FrozenSet, Iterable

from zoneinfo import ZoneInfo

__all__ = [
    "MarketCalendar",
    "MarketCalendarRegistry",
    "convert_timestamp",
    "get_market_calendar",
    "is_market_open",
    "normalize_timestamp",
    "to_utc",
]


def _ensure_iterable(values: Iterable[int] | None, *, default: Iterable[int]) -> FrozenSet[int]:
    if values is None:
        return frozenset(default)
    return frozenset(values)


@dataclass(frozen=True)
class MarketCalendar:
    """Defines opening hours and timezone for a market venue."""

    market: str
    timezone: str
    open_time: time = time(0, 0)
    close_time: time = time(23, 59, 59)
    weekend_closure: Iterable[int] | None = field(default_factory=lambda: {5, 6})
    holidays: Iterable[date] | None = None

    def __post_init__(self) -> None:
        if not self.market:
            raise ValueError("market must be a non-empty string")
        if not self.timezone:
            raise ValueError("timezone must be a non-empty string")
        object.__setattr__(self, "weekend_closure", _ensure_iterable(self.weekend_closure, default=(5, 6)))
        holidays = tuple(self.holidays or ())
        object.__setattr__(self, "holidays", holidays)

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_open(self, when: datetime) -> bool:
        local_time = convert_timestamp(when, self.market)
        if local_time.date() in self.holidays:
            return False
        if local_time.weekday() in self.weekend_closure:
            return False
        current_time = local_time.time()
        if self.open_time <= self.close_time:
            return self.open_time <= current_time <= self.close_time
        # Overnight sessions (e.g. futures)
        return current_time >= self.open_time or current_time <= self.close_time


_DEFAULT_CALENDARS: Dict[str, MarketCalendar] = {
    "BINANCE": MarketCalendar(market="BINANCE", timezone="UTC"),
    "NYSE": MarketCalendar(
        market="NYSE",
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
    ),
    "CME": MarketCalendar(
        market="CME",
        timezone="America/Chicago",
        open_time=time(17, 0),
        close_time=time(16, 0),
        weekend_closure=(5, 6),
    ),
}


class MarketCalendarRegistry:
    """Registry storing market calendars used throughout the codebase."""

    def __init__(self) -> None:
        self._calendars: Dict[str, MarketCalendar] = dict(_DEFAULT_CALENDARS)

    def register(self, calendar: MarketCalendar) -> None:
        self._calendars[calendar.market.upper()] = calendar

    def get(self, market: str) -> MarketCalendar:
        key = market.upper()
        if key not in self._calendars:
            raise KeyError(f"Unknown market calendar: {market}")
        return self._calendars[key]


_registry = MarketCalendarRegistry()


def get_market_calendar(market: str) -> MarketCalendar:
    """Return the configured calendar for the given market."""

    return _registry.get(market)


def convert_timestamp(ts: datetime, market: str) -> datetime:
    """Convert a timestamp to the timezone of a market."""

    calendar = get_market_calendar(market)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(calendar.tzinfo())


def to_utc(ts: datetime) -> datetime:
    """Ensure a timestamp is expressed in UTC."""

    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def normalize_timestamp(value: datetime | float | int, *, market: str | None = None) -> datetime:
    """Normalise raw timestamp inputs to a timezone-aware UTC datetime."""

    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = value
    else:  # pragma: no cover - defensive path
        raise TypeError("Unsupported timestamp type")

    if market is not None:
        dt = convert_timestamp(dt, market).astimezone(timezone.utc)
    else:
        dt = to_utc(dt)
    return dt


def is_market_open(ts: datetime, market: str) -> bool:
    """Return whether the market is open at the specified timestamp."""

    calendar = get_market_calendar(market)
    return calendar.is_open(ts)

