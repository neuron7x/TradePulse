"""Shared time-series persistence abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, Sequence


@dataclass(slots=True)
class TimeSeriesPoint:
    """Canonical representation of a metric or price observation."""

    timestamp: datetime
    values: dict[str, float]
    tags: dict[str, str] | None = None


class TimeSeriesAdapter(Protocol):
    """Protocol implemented by all concrete time-series adaptors."""

    def write_points(self, table: str, points: Sequence[TimeSeriesPoint]) -> int:
        """Persist points and return the number of inserted rows."""

    def read_points(
        self,
        table: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterable[TimeSeriesPoint]:
        """Stream points from storage."""

    def close(self) -> None:
        """Release any underlying resources."""
