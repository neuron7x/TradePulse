"""Shared time-series persistence abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable, Protocol, Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def sanitize_identifier(identifier: str) -> str:
    """Validate that *identifier* is safe for interpolation in SQL strings."""

    parts = [part for part in identifier.split(".") if part]
    if not parts:
        msg = "Identifier must not be empty"
        raise ValueError(msg)
    for part in parts:
        if not _IDENTIFIER.fullmatch(part):
            msg = f"Invalid identifier segment: {part!r}"
            raise ValueError(msg)
    return ".".join(parts)


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
