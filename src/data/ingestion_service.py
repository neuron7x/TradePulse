"""Data ingestion orchestration with caching helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from core.data.backfill import CacheKey, CacheRegistry, normalise_index
from core.data.catalog import normalize_symbol, normalize_venue
from core.data.ingestion import DataIngestor
from core.data.models import InstrumentType, PriceTick as Ticker
from core.data.quality_control import QualityReport, validate_and_quarantine
from core.data.validation import (
    TimeSeriesValidationConfig,
    TimeSeriesValidationError,
    ValueColumnConfig,
)
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
        integrity_validator: "TickFrameIntegrityValidator" | None = None,
    ) -> None:
        self._ingestor = data_ingestor or DataIngestor()
        self._registry = cache_registry or CacheRegistry()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._metadata: dict[CacheKey, CacheEntrySnapshot] = {}
        self._quality_reports: dict[CacheKey, QualityReport] = {}
        self._integrity_validator = integrity_validator or TickFrameIntegrityValidator()

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

        frame, _ = self.cache_ticks(
            records,
            layer=layer,
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
            market=market,
            instrument_type=instrument_type,
        )
        return frame

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
    ) -> tuple[pd.DataFrame, QualityReport]:
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
    ) -> tuple[pd.DataFrame, QualityReport]:
        """Store ``frame`` in the requested cache layer and update metadata.

        The returned tuple contains the cached frame (with anomalies removed) and
        the corresponding :class:`QualityReport` describing any quarantined
        rows.
        """

        if not timeframe or not timeframe.strip():
            raise ValueError("timeframe must be a non-empty string")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("frame must use a DatetimeIndex")

        normalized = frame.copy()
        if not normalized.empty:
            normalized = normalise_index(normalized, market=market).sort_index()

        cleaned, report = self._integrity_validator.validate(
            normalized,
            layer=layer,
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
        )
        key = self._build_key(layer, symbol, venue, timeframe, instrument_type)
        cache = self._registry.cache_for(layer)
        cache.put(key, cleaned)
        cached = cache.get(key)
        snapshot = self._build_snapshot(key, cached)
        self._metadata[key] = snapshot
        self._quality_reports[key] = report
        return cached, report

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

    def quality_report_for(
        self,
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
        instrument_type: InstrumentType = InstrumentType.SPOT,
    ) -> QualityReport | None:
        """Return the cached :class:`QualityReport` for the given key, if any."""

        key = self._build_key(layer, symbol, venue, timeframe, instrument_type)
        return self._quality_reports.get(key)

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
        self._quality_reports.clear()

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


class DataIntegrityError(ValueError):
    """Raised when cached datasets fail validation or integrity guarantees."""


class TickFrameIntegrityValidator:
    """Validate tick frames before they are persisted in the ingestion cache."""

    def __init__(
        self,
        *,
        timestamp_column: str = "timestamp",
        timezone: str = "UTC",
        anomaly_threshold: float = 6.0,
        anomaly_window: int = 20,
        price_column: str | None = "price",
    ) -> None:
        self._timestamp_column = timestamp_column
        self._timezone = timezone
        self._anomaly_threshold = anomaly_threshold
        self._anomaly_window = anomaly_window
        self._price_column = price_column

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        layer: str,
        symbol: str,
        venue: str,
        timeframe: str,
    ) -> tuple[pd.DataFrame, QualityReport]:
        """Validate ``frame`` and return a normalised copy and quality report."""

        config = self._build_config(frame, timeframe=timeframe, layer=layer)
        prepared = frame.reset_index().rename(
            columns={frame.index.name or "index": self._timestamp_column}
        )
        try:
            report = validate_and_quarantine(
                prepared,
                config,
                threshold=self._anomaly_threshold,
                window=self._anomaly_window,
                price_column=self._select_price_column(config),
            )
        except TimeSeriesValidationError as exc:
            raise DataIntegrityError(str(exc)) from exc

        def _to_index(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            if self._timestamp_column in result.columns:
                result = result.set_index(self._timestamp_column)
            result.index.name = frame.index.name
            return result.sort_index()

        cleaned = _to_index(report.clean)
        cleaned = cleaned.sort_index()
        return cleaned, QualityReport(
            clean=cleaned,
            quarantined=_to_index(report.quarantined),
            duplicates=_to_index(report.duplicates),
            spikes=_to_index(report.spikes),
        )

    def _build_config(
        self,
        frame: pd.DataFrame,
        *,
        timeframe: str,
        layer: str,
    ) -> TimeSeriesValidationConfig:
        frequency = None if layer == "raw" else self._parse_frequency(timeframe)
        value_columns: list[ValueColumnConfig] = []
        for column in frame.columns:
            series = frame[column]
            numeric_values = (
                series
                if pd.api.types.is_numeric_dtype(series.dtype)
                else pd.to_numeric(series, errors="coerce")
            )
            if series.isna().any():
                raise DataIntegrityError(f"{column} contains NaN values")
            if numeric_values.isna().any():
                raise DataIntegrityError(f"{column} contains non-numeric values")
            if not np.isfinite(numeric_values.to_numpy(copy=False)).all():
                raise DataIntegrityError(f"{column} contains non-finite values")
            value_columns.append(
                ValueColumnConfig(
                    name=column,
                    dtype=str(numeric_values.dtype),
                    nullable=False,
                )
            )

        return TimeSeriesValidationConfig(
            timestamp_column=self._timestamp_column,
            value_columns=value_columns,
            frequency=frequency,
            require_timezone=self._timezone,
            allow_extra_columns=True,
        )

    def _parse_frequency(self, timeframe: str) -> pd.Timedelta | None:
        trimmed = timeframe.strip()
        if not trimmed:
            return None
        try:
            return pd.to_timedelta(trimmed)
        except (TypeError, ValueError):
            return None

    def _select_price_column(self, config: TimeSeriesValidationConfig) -> str:
        candidates = [column.name for column in config.value_columns if column.name]
        if self._price_column and self._price_column in candidates:
            return self._price_column
        if candidates:
            return candidates[0]
        raise DataIntegrityError("No value columns available for anomaly detection")


__all__ = [
    "CacheEntrySnapshot",
    "DataIngestionCacheService",
    "DataIntegrityError",
    "TickFrameIntegrityValidator",
]
