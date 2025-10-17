"""Time utilities centralising timezone handling and conversions."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

UTC = timezone.utc


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def localize_to(ts: datetime, tz: str) -> datetime:
    return ensure_utc(ts).astimezone(pd.Timestamp.now(tz).tzinfo)

