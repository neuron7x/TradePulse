# SPDX-License-Identifier: MIT
"""Canonical market data models used across TradePulse.

The platform expects a single, strongly typed representation for all market
data payloads (ticks, OHLCV bars, aggregates) so downstream components can rely
on consistent validation semantics.  The models below are implemented with
``pydantic`` which provides strict runtime validation while keeping convenient
helpers for legacy construction patterns.

All timestamps are normalised to UTC, numeric values are coerced to ``Decimal``
and validated to avoid silent precision loss, and instrument metadata is shared
across every payload variant.  ``Ticker`` remains an alias for ``PriceTick`` to
preserve backwards compatibility with existing ingestion pipelines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, StrictStr, root_validator

try:  # pragma: no cover - runtime feature detection
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - executed on pydantic < 2
    ConfigDict = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime feature detection
    from pydantic import field_validator, field_serializer
except ImportError:  # pragma: no cover - executed on pydantic < 2
    field_validator = None  # type: ignore[assignment]
    field_serializer = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime feature detection
    from pydantic import validator as v1_validator
except ImportError:  # pragma: no cover - executed on pydantic >= 2
    v1_validator = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime feature detection
    from pydantic import model_validator
except ImportError:  # pragma: no cover - executed on pydantic < 2
    model_validator = None  # type: ignore[assignment]

if v1_validator is None:  # pragma: no cover - type-checking helper
    def v1_validator(*_: Any, **__: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("Pydantic v1 validator unavailable")

__all__ = [
    "AggregateMetric",
    "DataKind",
    "InstrumentType",
    "MarketDataPoint",
    "MarketMetadata",
    "OHLCVBar",
    "PriceTick",
    "Ticker",
]


class InstrumentType(str, Enum):
    """Enumerates the supported instrument categories."""

    SPOT = "spot"
    FUTURES = "futures"


class DataKind(str, Enum):
    """Enumerates the supported market data granularities."""

    TICK = "tick"
    OHLCV = "ohlcv"
    AGGREGATE = "aggregate"


def _to_decimal(value: Union[Decimal, float, int, str]) -> Decimal:
    """Convert arbitrary numeric inputs to ``Decimal`` safely."""

    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # bool is a subclass of int – avoid silent bugs
        raise TypeError("boolean values are not valid decimal inputs")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"Unable to convert {value!r} to Decimal") from exc


class _FrozenModel(BaseModel):
    """Base configuration shared by immutable market data models."""

    if ConfigDict is not None:  # pragma: no branch - simple feature flag
        model_config = ConfigDict(
            allow_mutation=False,
            str_strip_whitespace=True,
            extra="forbid",
            use_enum_values=False,
        )

        if field_serializer is not None:
            @field_serializer("*", when_used="json")
            def _serialize_decimal(cls, value: Any) -> Any:
                if isinstance(value, Decimal):
                    return str(value)
                return value
    else:
        # ``Config`` keeps backwards compatibility with Pydantic 1.x without
        # triggering the deprecation warning emitted by v2 once it is removed.
        class Config:  # type: ignore[too-many-ancestors]
            allow_mutation = False
            anystr_strip_whitespace = True
            extra = "forbid"
            use_enum_values = False
            json_encoders = {Decimal: str}


class MarketMetadata(_FrozenModel):
    """Common metadata shared by all market data payloads."""

    symbol: StrictStr = Field(..., min_length=1, description="Instrument symbol, e.g. BTCUSD")
    venue: StrictStr = Field(..., min_length=1, description="Market venue identifier")
    instrument_type: InstrumentType = Field(
        default=InstrumentType.SPOT,
        description="Instrument category (spot or futures)",
    )

    if field_validator is not None:
        @field_validator("symbol", "venue", mode="after")
        def _ensure_non_empty(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("value must be a non-empty string")
            return stripped
    else:
        @v1_validator("symbol", "venue")  # type: ignore[misc]
        def _ensure_non_empty(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("value must be a non-empty string")
            return stripped


class MarketDataPoint(_FrozenModel):
    """Base class for all market data records."""

    metadata: MarketMetadata
    timestamp: datetime
    kind: DataKind

    if field_validator is not None:
        @field_validator("timestamp", mode="before")
        def _coerce_timestamp(cls, value: Union[datetime, float, int]) -> datetime:
            if isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            elif isinstance(value, datetime):
                dt = value
            else:  # pragma: no cover - defensive guard
                raise TypeError("timestamp must be datetime or epoch seconds")

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt

        @field_validator("kind", mode="before")
        def _ensure_kind(cls, value: Union[None, DataKind, str]) -> DataKind:
            if value is None:
                raise ValueError("kind must be provided")
            if isinstance(value, DataKind):
                return value
            return DataKind(str(value))
    else:
        @v1_validator("timestamp", pre=True)  # type: ignore[misc]
        def _coerce_timestamp(cls, value: Union[datetime, float, int]) -> datetime:
            if isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            elif isinstance(value, datetime):
                dt = value
            else:  # pragma: no cover - defensive guard
                raise TypeError("timestamp must be datetime or epoch seconds")

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt

        @v1_validator("kind", pre=True)  # type: ignore[misc]
        def _ensure_kind(cls, value: Union[None, DataKind, str]) -> DataKind:
            if value is None:
                raise ValueError("kind must be provided")
            if isinstance(value, DataKind):
                return value
            return DataKind(str(value))

    @property
    def symbol(self) -> str:
        return self.metadata.symbol

    @property
    def venue(self) -> str:
        return self.metadata.venue

    @property
    def instrument_type(self) -> InstrumentType:
        return self.metadata.instrument_type

    @property
    def ts(self) -> float:
        """Return the timestamp as epoch seconds."""

        return self.timestamp.timestamp()


class PriceTick(MarketDataPoint):
    """Tick-level price update."""

    price: Decimal = Field(..., description="Last traded price")
    volume: Decimal = Field(default=Decimal("0"), description="Trade volume at the tick")
    trade_id: Optional[str] = Field(default=None, description="Exchange trade identifier")
    kind: Literal[DataKind.TICK] = DataKind.TICK

    if field_validator is not None:
        @field_validator("price", mode="before")
        def _coerce_price(cls, value: Union[Decimal, float, int, str, None]) -> Decimal:
            if value is None:
                raise ValueError("price must be provided")
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @field_validator("volume", mode="before")
        def _coerce_volume(cls, value: Union[Decimal, float, int, str, None]) -> Decimal:
            if value is None:
                return Decimal("0")
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @field_validator("price", "volume", mode="after")
        def _validate_non_negative(cls, value: Decimal) -> Decimal:
            if value < 0:
                raise ValueError("numeric values must be non-negative")
            return value

        @field_validator("trade_id", mode="after")
        def _strip_trade_id(cls, value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            stripped = value.strip()
            return stripped or None
    else:
        @v1_validator("price", pre=True)  # type: ignore[misc]
        def _coerce_price(cls, value: Union[Decimal, float, int, str, None]) -> Decimal:
            if value is None:
                raise ValueError("price must be provided")
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @v1_validator("volume", pre=True)  # type: ignore[misc]
        def _coerce_volume(cls, value: Union[Decimal, float, int, str, None]) -> Decimal:
            if value is None:
                return Decimal("0")
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @v1_validator("price", "volume")  # type: ignore[misc]
        def _validate_non_negative(cls, value: Decimal) -> Decimal:
            if value < 0:
                raise ValueError("numeric values must be non-negative")
            return value

        @v1_validator("trade_id")  # type: ignore[misc]
        def _strip_trade_id(cls, value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            stripped = value.strip()
            return stripped or None

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        venue: str,
        price: Union[Decimal, float, int, str],
        timestamp: Union[datetime, float, int],
        volume: Union[Decimal, float, int, str, None] = None,
        instrument_type: InstrumentType = InstrumentType.SPOT,
        trade_id: Optional[str] = None,
    ) -> "PriceTick":
        """Factory helper that builds the metadata block for convenience."""

        meta = MarketMetadata(symbol=symbol, venue=venue, instrument_type=instrument_type)
        return cls(
            metadata=meta,
            timestamp=timestamp,
            price=price,
            volume=volume,
            trade_id=trade_id,
        )


class OHLCVBar(MarketDataPoint):
    """OHLCV bar representing aggregated price information."""

    open: Decimal = Field(..., description="Opening price")
    high: Decimal = Field(..., description="Highest price in the interval")
    low: Decimal = Field(..., description="Lowest price in the interval")
    close: Decimal = Field(..., description="Closing price")
    volume: Decimal = Field(..., description="Total traded volume")
    interval_seconds: int = Field(..., gt=0, description="Bar interval in seconds")
    kind: Literal[DataKind.OHLCV] = DataKind.OHLCV

    if field_validator is not None:
        @field_validator("open", "high", "low", "close", "volume", mode="before")
        def _coerce_decimal_values(cls, value: Union[Decimal, float, int, str]) -> Decimal:
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @field_validator("open", "high", "low", "close", "volume", mode="after")
        def _validate_non_negative(cls, value: Decimal) -> Decimal:
            if value < 0:
                raise ValueError("OHLCV values must be non-negative")
            return value
    else:
        @v1_validator("open", "high", "low", "close", "volume", pre=True)  # type: ignore[misc]
        def _coerce_decimal_values(cls, value: Union[Decimal, float, int, str]) -> Decimal:
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc

        @v1_validator("open", "high", "low", "close", "volume")  # type: ignore[misc]
        def _validate_non_negative(cls, value: Decimal) -> Decimal:
            if value < 0:
                raise ValueError("OHLCV values must be non-negative")
            return value

    if model_validator is None:
        @root_validator
        def _validate_price_relationships(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            low = values.get("low")
            high = values.get("high")
            open_price = values.get("open")
            close_price = values.get("close")
            if low is not None and high is not None and high < low:
                raise ValueError("high price must be greater or equal to low price")
            if low is not None and high is not None:
                if open_price is not None and not (low <= open_price <= high):
                    raise ValueError("open price must lie between low and high")
                if close_price is not None and not (low <= close_price <= high):
                    raise ValueError("close price must lie between low and high")
            return values

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        low = self.low
        high = self.high
        open_price = self.open
        close_price = self.close
        if high < low:
            raise ValueError("high price must be greater or equal to low price")
        if open_price < low or open_price > high:
            raise ValueError("open price must lie between low and high")
        if close_price < low or close_price > high:
            raise ValueError("close price must lie between low and high")


class AggregateMetric(MarketDataPoint):
    """Generic aggregated value produced from raw market data."""

    metric: StrictStr = Field(..., min_length=1, description="Metric name")
    value: Decimal = Field(..., description="Metric value")
    window_seconds: int = Field(..., gt=0, description="Window size in seconds")
    kind: Literal[DataKind.AGGREGATE] = DataKind.AGGREGATE

    if field_validator is not None:
        @field_validator("metric", mode="after")
        def _validate_metric(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("metric must be a non-empty string")
            return stripped

        @field_validator("value", mode="before")
        def _coerce_value(cls, value: Union[Decimal, float, int, str]) -> Decimal:
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc
    else:
        @v1_validator("metric")  # type: ignore[misc]
        def _validate_metric(cls, value: str) -> str:
            stripped = value.strip()
            if not stripped:
                raise ValueError("metric must be a non-empty string")
            return stripped

        @v1_validator("value", pre=True)  # type: ignore[misc]
        def _coerce_value(cls, value: Union[Decimal, float, int, str]) -> Decimal:
            try:
                return _to_decimal(value)
            except TypeError as exc:  # pragma: no cover - propagated via ValidationError
                raise ValueError(str(exc)) from exc


# Backwards compatibility export ---------------------------------------------------------

Ticker = PriceTick
