"""Trading calendar utilities for deterministic backtests.

The calendar tracks regular trading hours, exchange holidays, and ad-hoc
session overrides while handling timezone daylight-saving transitions. The
intent is to provide deterministic scheduling utilities for backtests that need
to respect venue trading windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

__all__ = ["SessionHours", "MarketCalendar"]


@dataclass(frozen=True)
class SessionHours:
    """Represents the open and close time (inclusive/exclusive) of a session."""

    open: time
    close: time

    @classmethod
    def from_value(cls, value: SessionHours | Sequence[time]) -> SessionHours:
        if isinstance(value, SessionHours):
            return value
        if (
            isinstance(value, Sequence)
            and len(value) == 2
            and isinstance(value[0], time)
            and isinstance(value[1], time)
        ):
            return cls(open=value[0], close=value[1])
        raise TypeError(
            "Session hours must be SessionHours or a (open, close) pair of time objects"
        )


class MarketCalendar:
    """Lightweight trading calendar with DST-aware session calculations."""

    def __init__(
        self,
        timezone: str,
        regular_hours: Mapping[int, SessionHours | Sequence[time]],
        *,
        holidays: Iterable[date] | None = None,
        special_sessions: Mapping[date, SessionHours | Sequence[time]] | None = None,
    ) -> None:
        self._tz = ZoneInfo(timezone)
        self._regular_hours = {
            int(weekday): SessionHours.from_value(session)
            for weekday, session in regular_hours.items()
        }
        self._holidays = {d for d in (holidays or [])}
        self._special_sessions = {
            session_date: SessionHours.from_value(session)
            for session_date, session in (special_sessions or {}).items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_open(self, timestamp: datetime) -> bool:
        """Return True if ``timestamp`` falls inside a trading session."""

        local_ts = self._localize(timestamp)
        session = self._session_for_date(local_ts.date())
        if session is None:
            return False

        open_dt = self._combine(local_ts.date(), session.open)
        close_dt = self._combine(local_ts.date(), session.close)
        if close_dt <= open_dt:
            close_dt += timedelta(days=1)
        return open_dt <= local_ts < close_dt

    def next_open(self, timestamp: datetime) -> datetime:
        """Return the next session open strictly after ``timestamp``."""

        search_dt = self._localize(timestamp)
        while True:
            session = self._session_for_date(search_dt.date())
            if session is not None:
                open_dt = self._combine(search_dt.date(), session.open)
                close_dt = self._combine(search_dt.date(), session.close)
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                if search_dt < open_dt:
                    return open_dt
                if search_dt < close_dt:
                    search_dt = close_dt + timedelta(microseconds=1)
                    continue
            search_dt = self._combine(search_dt.date() + timedelta(days=1), time(0, 0))

    def previous_close(self, timestamp: datetime) -> datetime:
        """Return the most recent session close at or before ``timestamp``."""

        search_dt = self._localize(timestamp)
        while True:
            session = self._session_for_date(search_dt.date())
            if session is not None:
                open_dt = self._combine(search_dt.date(), session.open)
                close_dt = self._combine(search_dt.date(), session.close)
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                if search_dt >= close_dt:
                    return close_dt
                if search_dt >= open_dt:
                    return close_dt
            search_dt = self._combine(
                search_dt.date() - timedelta(days=1), time(23, 59, 59, 999999)
            )

    def sessions_between(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Enumerate sessions intersecting the inclusive ``[start, end]`` window."""

        if end < start:
            raise ValueError("end must be greater than or equal to start")

        start_local = self._localize(start)
        end_local = self._localize(end)
        sessions: list[tuple[datetime, datetime]] = []

        cursor_date = start_local.date()
        while True:
            session = self._session_for_date(cursor_date)
            if session is not None:
                open_dt = self._combine(cursor_date, session.open)
                close_dt = self._combine(cursor_date, session.close)
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                if close_dt < start_local:
                    pass
                elif open_dt > end_local:
                    break
                else:
                    sessions.append((open_dt, close_dt))
                    if close_dt >= end_local:
                        break

            cursor_midnight = self._combine(cursor_date, time(0, 0))
            if cursor_midnight > end_local:
                break
            cursor_date += timedelta(days=1)

        return sessions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _session_for_date(self, session_date: date) -> SessionHours | None:
        if session_date in self._holidays:
            return None
        if session_date in self._special_sessions:
            return self._special_sessions[session_date]
        weekday = session_date.weekday()
        return self._regular_hours.get(weekday)

    def _localize(self, timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=self._tz)
        return timestamp.astimezone(self._tz)

    def _combine(self, session_date: date, when: time) -> datetime:
        return datetime.combine(session_date, when, tzinfo=self._tz)
