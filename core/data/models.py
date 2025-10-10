# SPDX-License-Identifier: MIT
"""Canonical market data models used across TradePulse.

The project historically relied on light-weight ``dataclass`` definitions for
tick data.  As the ingestion stack grew, the lack of a shared contract for
different market data shapes (ticks, OHLCV bars, aggregated metrics and
derivatives) started to surface bugs.  This module introduces a strict
Pydantic-powered schema that normalises how market data travels through the
system while still remaining light-weight enough for analytics workloads.

The goal of this module is twofold:

* Provide strongly-typed models with validation for common market payloads.
* Encode instrument metadata (spot vs futures) in a consistent way.

The models are frozen to make accidental mutation impossible.  All
timestamps are normalised to timezone-aware ``datetime`` instances using the
UTC timezone by default.  Consumers who need the exchange-local timestamp can
use the helper utilities from :mod:`core.data.timeutils`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StrictPrice = Annotated[
    Decimal,
    Field(
        ge=Decimal("0"),
        max_digits=28,
        decimal_places=12,
        description="Non-negative monetary value",
    ),
]
StrictVolume = Annotated[
    Decimal,
    Field(
        ge=Decimal("0"),
        max_digits=28,
        decimal_places=12,
        description="Non-negative traded size",
    ),
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


class MarketMetadata(BaseModel):
    """Common metadata shared by all market data payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    symbol: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    instrument_type: InstrumentType = InstrumentType.SPOT


class MarketDataPoint(BaseModel):
    """Base class for all market data records."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    metadata: MarketMetadata
    timestamp: datetime = Field(description="UTC timestamp of the data point")
    kind: DataKind

    @field_validator("timestamp", mode="after")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

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
        return self.timestamp.timestamp()


class PriceTick(MarketDataPoint):
    """Tick-level price update."""

    kind: Literal[DataKind.TICK] = DataKind.TICK
    price: StrictPrice
    volume: StrictVolume = Field(default=Decimal("0"))
    trade_id: Optional[str] = Field(default=None, description="Exchange trade identifier")

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        venue: str,
        price: Union[Decimal, float, str],
        timestamp: Union[datetime, float, int],
        volume: Union[Decimal, float, str, None] = None,
        instrument_type: InstrumentType = InstrumentType.SPOT,
        trade_id: Optional[str] = None,
    ) -> "PriceTick":
        """Factory helper that builds the metadata block for convenience."""

        meta = MarketMetadata(symbol=symbol, venue=venue, instrument_type=instrument_type)
        vol = Decimal("0") if volume is None else Decimal(str(volume))
        price_decimal = Decimal(str(price))
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:  # pragma: no cover - defensive path
            raise TypeError("timestamp must be datetime or unix epoch")
        return cls(
            metadata=meta,
            timestamp=dt,
            price=price_decimal,
            volume=vol,
            trade_id=trade_id,
        )


class OHLCVBar(MarketDataPoint):
    """OHLCV bar representing aggregated price information."""

    kind: Literal[DataKind.OHLCV] = DataKind.OHLCV
    open: StrictPrice
    high: StrictPrice
    low: StrictPrice
    close: StrictPrice
    volume: StrictVolume
    interval_seconds: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_bar(self) -> "OHLCVBar":
        if self.high < self.low:
            raise ValueError("high price must be greater or equal to low price")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open price must lie between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close price must lie between low and high")
        return self


class AggregateMetric(MarketDataPoint):
    """Generic aggregated value produced from raw market data."""

    kind: Literal[DataKind.AGGREGATE] = DataKind.AGGREGATE
    metric: str = Field(min_length=1)
    value: Decimal
    window_seconds: int = Field(ge=1)


# Backwards compatibility export ------------------------------------------------

# ``Ticker`` used to be a ``dataclass`` consumed widely across the code base.
# Re-export the new ``PriceTick`` under the legacy name so dependent modules
# continue to work while benefitting from the stricter validation layer.
Ticker = PriceTick


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
