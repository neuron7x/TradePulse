"""Incremental backfill helpers with hierarchical caching.

The ingestion layer maintains three logical cache tiers:

``raw``
    Tick-by-tick payloads before any transformation.  Useful for replays.
``ohlcv``
    Aggregated bars aligned to a canonical calendar.
``features``
    Derived indicator buffers that are expensive to recompute.

All caches expose the same interface so they can be stacked together.  Gap
detection is central to the incremental workflow: given an expected sampling
frequency the planner identifies missing ranges and only requests those slices
from the upstream provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC
from typing import List, MutableMapping, Optional

import pandas as pd

from core.data.timeutils import normalize_timestamp


@dataclass(frozen=True)
class CacheKey:
    """Compound cache key covering symbol, venue and timeframe."""

    layer: str
    symbol: str
    venue: str
    timeframe: str


@dataclass
class CacheEntry:
    """Cache metadata and payload."""

    frame: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp

    def slice(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
        view = self.frame
        if start is not None:
            view = view[view.index >= start]
        if end is not None:
            view = view[view.index <= end]
        return view.copy()


class LayerCache:
    """In-memory cache implementing the shared ingestion cache protocol."""

    def __init__(self) -> None:
        self._entries: MutableMapping[CacheKey, CacheEntry] = {}

    def put(self, key: CacheKey, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("Cache payload must be indexed by pd.DatetimeIndex")
        start = frame.index.min()
        end = frame.index.max()
        self._entries[key] = CacheEntry(frame=frame.copy(), start=start, end=end)

    def get(
        self,
        key: CacheKey,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        entry = self._entries.get(key)
        if entry is None:
            return pd.DataFrame()
        return entry.slice(start, end)

    def coverage(self, key: CacheKey) -> Optional[pd.Interval]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return pd.Interval(entry.start, entry.end, closed="both")


@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("Gap requires start < end")


def detect_gaps(
    expected_index: pd.DatetimeIndex,
    existing_index: pd.DatetimeIndex,
) -> List[Gap]:
    """Return gaps between ``expected_index`` and ``existing_index``."""

    missing = expected_index.difference(existing_index)
    if missing.empty:
        return []

    gaps: List[Gap] = []
    start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts - prev > expected_index.freq:
            gaps.append(Gap(start=start, end=prev + expected_index.freq))
            start = ts
        prev = ts
    gaps.append(Gap(start=start, end=prev + expected_index.freq))
    return gaps


@dataclass
class BackfillPlan:
    """Description of the windows that need to be requested."""

    gaps: List[Gap] = field(default_factory=list)
    covered: Optional[pd.Interval] = None

    @property
    def is_full_refresh(self) -> bool:
        return not self.covered and bool(self.gaps)


class GapFillPlanner:
    """Analyse cache coverage and produce backfill plans."""

    def __init__(self, cache: LayerCache) -> None:
        self._cache = cache

    def plan(
        self,
        key: CacheKey,
        *,
        expected_index: pd.DatetimeIndex,
    ) -> BackfillPlan:
        coverage = self._cache.coverage(key)
        existing = self._cache.get(key)
        if existing.empty:
            return BackfillPlan(gaps=[Gap(start=expected_index[0], end=expected_index[-1] + expected_index.freq)])
        gaps = detect_gaps(expected_index, existing.index)
        return BackfillPlan(gaps=gaps, covered=coverage)

    def apply(
        self,
        key: CacheKey,
        frame: pd.DataFrame,
    ) -> None:
        if frame.empty:
            return
        current = self._cache.get(key)
        if current.empty:
            self._cache.put(key, frame)
            return
        combined = pd.concat([current, frame])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        self._cache.put(key, combined)


class CacheRegistry:
    """Facade aggregating raw/ohlcv/feature caches."""

    def __init__(self) -> None:
        self.raw = LayerCache()
        self.ohlcv = LayerCache()
        self.features = LayerCache()

    def cache_for(self, layer: str) -> LayerCache:
        if layer not in {"raw", "ohlcv", "features"}:
            raise ValueError(f"Unknown cache layer: {layer}")
        return getattr(self, layer)


def normalise_index(frame: pd.DataFrame, *, market: Optional[str] = None) -> pd.DataFrame:
    """Ensure the index is tz-aware and normalised through ``normalize_timestamp``."""

    if frame.empty:
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex")
    normalized = [normalize_timestamp(ts.to_pydatetime(), market=market) for ts in frame.index]
    result = frame.copy()
    result.index = pd.DatetimeIndex(normalized, tz=UTC)
    return result


__all__ = [
    "BackfillPlan",
    "CacheEntry",
    "CacheKey",
    "CacheRegistry",
    "Gap",
    "GapFillPlanner",
    "LayerCache",
    "detect_gaps",
    "normalise_index",
]

