"""Data ingestion orchestration with caching helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import pandas as pd

from core.data.backfill import CacheKey, CacheRegistry, normalise_index
from core.data.catalog import normalize_symbol, normalize_venue
from core.data.ingestion import DataIngestor
from core.data.models import InstrumentType, PriceTick as Ticker
from interfaces.ingestion import DataIngestionService

UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class CacheEntrySnapshot:
    """Summary of a cached dataset stored in the ingestion cache."""

    key: CacheKey
    rows: int
    start: datetime | None
    end: datetime | None
    last_updated: datetime


class DataIngestionCacheService:
    """Coordinate ingestion flows and maintain in-memory caches.

    The service wraps :class:`~core.data.ingestion.DataIngestor` (or any
    implementation of :class:`interfaces.ingestion.DataIngestionService`) and
    records the resulting data frames in a :class:`CacheRegistry`.  This allows
    callers to ingest historical files or pre-built tick buffers, hydrate
    Pandas frames, and serve subsequent requests directly from the cache
    without touching the underlying storage again.
    """

    def __init__(
        self,
        *,
        data_ingestor: DataIngestionService | None = None,
        cache_registry: CacheRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._ingestor = data_ingestor or DataIngestor()
        self._registry = cache_registry or CacheRegistry()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._metadata: dict[CacheKey, CacheEntrySnapshot] = {}

    # ------------------------------------------------------------------
    # Public API
    def ingest_csv(
        self,
        path: str,
        *,
        symbol: str,
        venue: str,
        timeframe: str,
        instrument_type: InstrumentType = InstrumentType.SPOT,
        market: str | None = None,
        required_fields: Iterable[str] = ("ts", "price"),
        layer: str = "raw",
    ) -> pd.DataFrame:
        """Ingest a CSV file and cache the resulting tick frame."""

        records: list[Ticker] = []
        self._ingestor.historical_csv(
            path,
            records.append,
            required_fields=required_fields,
            symbol=symbol,
            venue=venue,
            instrument_type=instrument_type,
            market=market,
        )

        if not records:
            raise ValueError(f"No ticks were ingested from {path}")

        return self.cache_ticks(
            records,
            layer=layer,
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
            market=market,
            instrument_type=instrument_type,
        )

    def cache_ticks(
        self,
        ticks: Sequence[Ticker],
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        market: str | None = None,
        instrument_type: InstrumentType | None = None,
    ) -> pd.DataFrame:
        """Cache a sequence of ticks under the provided cache key."""

        if not ticks:
            raise ValueError("ticks must not be empty")
        if not timeframe or not timeframe.strip():
            raise ValueError("timeframe must be a non-empty string")
        resolved_type = instrument_type or ticks[0].instrument_type
        key = self._build_key(layer, symbol, venue, timeframe, resolved_type)
        if any(tick.symbol != key.symbol for tick in ticks):
            raise ValueError("All ticks must match the provided symbol")
        if any(tick.venue != key.venue for tick in ticks):
            raise ValueError("All ticks must match the provided venue")

        frame = self._ticks_to_frame(ticks)
        return self.cache_frame(
            frame,
            layer=layer,
            symbol=key.symbol,
            venue=key.venue,
            timeframe=key.timeframe,
            market=market,
            instrument_type=resolved_type,
        )

    def cache_frame(
        self,
        frame: pd.DataFrame,
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        market: str | None = None,
        instrument_type: InstrumentType = InstrumentType.SPOT,
    ) -> pd.DataFrame:
        """Store ``frame`` in the requested cache layer and update metadata."""

        if not timeframe or not timeframe.strip():
            raise ValueError("timeframe must be a non-empty string")
        if frame.empty:
            normalized = frame.copy()
        else:
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError("frame must use a DatetimeIndex")
            normalized = normalise_index(frame, market=market).sort_index()
        key = self._build_key(layer, symbol, venue, timeframe, instrument_type)
        cache = self._registry.cache_for(layer)
        cache.put(key, normalized)
        cached = cache.get(key)
        snapshot = self._build_snapshot(key, cached)
        self._metadata[key] = snapshot
        return cached

    def get_cached_frame(
        self,
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        instrument_type: InstrumentType = InstrumentType.SPOT,
    ) -> pd.DataFrame:
        """Return a cached frame optionally filtered by ``start``/``end``."""

        key = self._build_key(layer, symbol, venue, timeframe, instrument_type)
        cache = self._registry.cache_for(layer)
        start_ts = self._coerce_datetime(start)
        end_ts = self._coerce_datetime(end)
        return cache.get(key, start=start_ts, end=end_ts)

    def metadata_for(
        self,
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        instrument_type: InstrumentType = InstrumentType.SPOT,
    ) -> CacheEntrySnapshot | None:
        """Return cached metadata for the given key if present."""

        key = self._build_key(layer, symbol, venue, timeframe, instrument_type)
        return self._metadata.get(key)

    def cache_snapshot(self) -> list[CacheEntrySnapshot]:
        """Return metadata for all cached datasets ordered deterministically."""

        return sorted(
            self._metadata.values(),
            key=lambda entry: (entry.key.layer, entry.key.symbol, entry.key.venue, entry.key.timeframe),
        )

    def clear(self) -> None:
        """Reset the cache registry and forget all cached metadata."""

        self._registry = CacheRegistry()
        self._metadata.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    def _coerce_datetime(self, value: datetime | None) -> pd.Timestamp | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return pd.Timestamp(value)

    def _ticks_to_frame(self, ticks: Sequence[Ticker]) -> pd.DataFrame:
        index = pd.DatetimeIndex([tick.timestamp for tick in ticks])
        data = {
            "price": [float(tick.price) for tick in ticks],
            "volume": [float(tick.volume) for tick in ticks],
        }
        frame = pd.DataFrame(data, index=index)
        frame.index.name = "timestamp"
        return frame

    def _build_snapshot(self, key: CacheKey, frame: pd.DataFrame) -> CacheEntrySnapshot:
        if frame.empty:
            start = end = None
            rows = 0
        else:
            start = frame.index.min().to_pydatetime()
            end = frame.index.max().to_pydatetime()
            rows = int(frame.shape[0])
        timestamp = self._clock()
        return CacheEntrySnapshot(key=key, rows=rows, start=start, end=end, last_updated=timestamp)

    def _build_key(
        self,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        instrument_type: InstrumentType,
    ) -> CacheKey:
        canonical_symbol = normalize_symbol(symbol, instrument_type_hint=instrument_type)
        canonical_venue = normalize_venue(venue)
        return CacheKey(layer=layer, symbol=canonical_symbol, venue=canonical_venue, timeframe=timeframe)


__all__ = ["CacheEntrySnapshot", "DataIngestionCacheService"]
