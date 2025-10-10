# SPDX-License-Identifier: MIT
"""Centralised timezone and market calendar helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Sequence

import pandas as pd
from exchange_calendars import ExchangeCalendar, always_open, errors, get_calendar, resolve_alias
from zoneinfo import ZoneInfo

__all__ = [
    "MarketCalendar",
    "MarketCalendarRegistry",
    "convert_timestamp",
    "get_market_calendar",
    "get_timezone",
    "is_market_open",
    "normalize_timestamp",
    "to_utc",
    "validate_bar_alignment",
]


def _ensure_iterable(values: Iterable[int] | None, *, default: Iterable[int]) -> FrozenSet[int]:
    if values is None:
        return frozenset(default)
    return frozenset(values)


@lru_cache(maxsize=None)
def _load_exchange_calendar(name: str) -> ExchangeCalendar:
    """Resolve an ``exchange_calendars`` identifier into a calendar instance."""

    alias = name.upper()
    if alias in {"ALWAYS_OPEN", "24/7", "247"}:
        return always_open.AlwaysOpenCalendar()
    try:
        resolved = resolve_alias(alias)
    except errors.InvalidCalendarName:
        resolved = alias
    return get_calendar(resolved)


@dataclass(frozen=True)
class MarketCalendar:
    """Defines opening hours and timezone for a market venue."""

    market: str
    timezone: str | None = None
    open_time: time = time(0, 0)
    close_time: time = time(23, 59, 59)
    weekend_closure: Iterable[int] | None = field(default_factory=lambda: {5, 6})
    holidays: Iterable[date] | None = None
    calendar_name: str | None = None
    _calendar: ExchangeCalendar | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.market:
            raise ValueError("market must be a non-empty string")
        if self.calendar_name is not None:
            calendar = _load_exchange_calendar(self.calendar_name)
            object.__setattr__(self, "_calendar", calendar)
            calendar_tz = calendar.tz
            tz_value = getattr(calendar_tz, "key", None) or str(calendar_tz)
            object.__setattr__(self, "timezone", tz_value)
            default_weekend = tuple()
        else:
            if not self.timezone:
                raise ValueError("timezone must be a non-empty string")
            default_weekend = (5, 6)
        object.__setattr__(self, "weekend_closure", _ensure_iterable(self.weekend_closure, default=default_weekend))
        holidays = tuple(self.holidays or ())
        object.__setattr__(self, "holidays", holidays)

    @property
    def exchange_calendar(self) -> ExchangeCalendar | None:
        return self._calendar

    def tzinfo(self) -> ZoneInfo:
        if not self.timezone:
            raise ValueError("timezone has not been configured")
        return ZoneInfo(self.timezone)

    def is_open(self, when: datetime) -> bool:
        if self._calendar is not None:
            ts = _ensure_timestamp(when)
            return bool(self._calendar.is_open_on_minute(ts))
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


def _ensure_timestamp(value: datetime) -> pd.Timestamp:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


_BINANCE_CALENDAR = MarketCalendar(market="BINANCE", calendar_name="ALWAYS_OPEN", weekend_closure=())
_NYSE_CALENDAR = MarketCalendar(market="NYSE", calendar_name="XNYS")
_NASDAQ_CALENDAR = MarketCalendar(market="NASDAQ", calendar_name="XNAS")
_CME_CALENDAR = MarketCalendar(market="CME", calendar_name="CMES")

_DEFAULT_CALENDARS: Dict[str, MarketCalendar] = {
    "BINANCE": _BINANCE_CALENDAR,
    "ALWAYS_OPEN": _BINANCE_CALENDAR,
    "24/7": _BINANCE_CALENDAR,
    "247": _BINANCE_CALENDAR,
    "NYSE": _NYSE_CALENDAR,
    "XNYS": _NYSE_CALENDAR,
    "NASDAQ": _NASDAQ_CALENDAR,
    "XNAS": _NASDAQ_CALENDAR,
    "CME": _CME_CALENDAR,
    "CMES": _CME_CALENDAR,
}


class MarketCalendarRegistry:
    """Registry storing market calendars used throughout the codebase."""

    def __init__(self) -> None:
        self._calendars: Dict[str, MarketCalendar] = dict(_DEFAULT_CALENDARS)
        for calendar in _DEFAULT_CALENDARS.values():
            if calendar.calendar_name is not None:
                self._calendars.setdefault(calendar.calendar_name.upper(), calendar)

    def register(self, calendar: MarketCalendar) -> None:
        key = calendar.market.upper()
        self._calendars[key] = calendar
        if calendar.calendar_name is not None:
            alias = calendar.calendar_name.upper()
            self._calendars[alias] = calendar
            if alias == "ALWAYS_OPEN":
                self._calendars.setdefault("24/7", calendar)
                self._calendars.setdefault("247", calendar)

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


def get_timezone(name: str) -> ZoneInfo:
    """Resolve a timezone identifier to a :class:`ZoneInfo` instance."""

    try:
        return ZoneInfo(name)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown timezone identifier: {name}") from exc


def _as_utc_index(values: Sequence[datetime | pd.Timestamp]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return index


def validate_bar_alignment(
    timestamps: Sequence[datetime | pd.Timestamp],
    *,
    market: str,
    frequency: str | pd.Timedelta,
) -> None:
    """Ensure time-series bars align with the target market calendar and frequency.

    Parameters
    ----------
    timestamps:
        Sequence of timestamps expected to be sorted in chronological order.
    market:
        Market identifier used to resolve the associated calendar.
    frequency:
        Sampling frequency expressed as a pandas-compatible timedelta string or ``Timedelta``.
    """

    values = list(timestamps)
    if not values:
        return

    index = _as_utc_index(values)
    if not index.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")
    if index.has_duplicates:
        raise ValueError("timestamps contain duplicate entries")

    freq = pd.Timedelta(frequency)
    if freq <= pd.Timedelta(0):
        raise ValueError("frequency must be a positive duration")

    one_minute = pd.Timedelta(minutes=1)
    if freq % one_minute != pd.Timedelta(0):
        raise ValueError("frequency must be an integer number of minutes")

    calendar = get_market_calendar(market)
    exchange_calendar = calendar.exchange_calendar

    if exchange_calendar is None:
        expected = pd.date_range(start=index[0], end=index[-1], freq=freq, tz="UTC")
        if not index.equals(expected):
            raise ValueError("timestamps do not align with the requested frequency")
        return

    trading_minutes = exchange_calendar.minutes_in_range(index[0], index[-1])
    try:
        start_loc = trading_minutes.get_loc(index[0])
    except KeyError as exc:
        raise ValueError(f"{index[0]} is not a valid trading minute for {market}") from exc
    try:
        end_loc = trading_minutes.get_loc(index[-1])
    except KeyError as exc:
        raise ValueError(f"{index[-1]} is not a valid trading minute for {market}") from exc

    step = int(freq / one_minute)
    expected = trading_minutes[start_loc : end_loc + 1 : step]

    if len(expected) != len(index) or not index.equals(expected):
        missing = expected.difference(index)
        extra = index.difference(expected)
        missing_repr = [ts.isoformat() for ts in missing.to_pydatetime()]
        extra_repr = [ts.isoformat() for ts in extra.to_pydatetime()]
        raise ValueError(
            "timestamps do not align with the trading calendar; "
            f"missing={missing_repr}, extra={extra_repr}",
        ) from None

