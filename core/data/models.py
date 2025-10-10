# SPDX-License-Identifier: MIT
"""Canonical market data models used across TradePulse.

This module intentionally avoids heavyweight runtime dependencies so it can be
used from lightweight ingestion scripts as well as the core application.  The
previous iteration relied on ``pydantic`` for validation which introduced an
optional dependency that is not available in minimal environments (and broke
the test suite).  The implementation below keeps the strong validation
semantics but expresses the models as frozen ``dataclass`` structures.

The helpers mirror the behaviour that the rest of the codebase expects from the
``Ticker`` interface while adding dedicated classes for OHLCV bars and
aggregated metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional, Union

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
    # ``bool`` is a subclass of ``int`` – explicitly forbid it because it is
    # almost always a bug when passed as a price/volume value.
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid decimal inputs")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"Unable to convert {value!r} to Decimal") from exc


@dataclass(frozen=True)
class MarketMetadata:
    """Common metadata shared by all market data payloads."""

    symbol: str
    venue: str
    instrument_type: InstrumentType = InstrumentType.SPOT

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if not self.venue:
            raise ValueError("venue must be a non-empty string")


@dataclass(frozen=True)
class MarketDataPoint:
    """Base class for all market data records."""

    metadata: MarketMetadata
    timestamp: datetime
    kind: DataKind

    def __post_init__(self) -> None:
        ts = self.timestamp
        if not isinstance(ts, datetime):
            raise TypeError("timestamp must be a datetime instance")
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        object.__setattr__(self, "timestamp", ts)

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


@dataclass(frozen=True)
class PriceTick(MarketDataPoint):
    """Tick-level price update."""

    price: Decimal
    volume: Decimal = Decimal("0")
    trade_id: Optional[str] = None
    kind: DataKind = field(default=DataKind.TICK, init=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        price = _to_decimal(self.price)
        volume = _to_decimal(self.volume)
        if price < 0:
            raise ValueError("price must be non-negative")
        if volume < 0:
            raise ValueError("volume must be non-negative")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "volume", volume)

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
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            raise TypeError("timestamp must be datetime or unix epoch")
        vol_value: Union[Decimal, float, int, str]
        if volume is None:
            vol_value = Decimal("0")
        else:
            vol_value = volume
        return cls(
            metadata=meta,
            timestamp=dt,
            price=_to_decimal(price),
            volume=_to_decimal(vol_value),
            trade_id=trade_id,
        )


@dataclass(frozen=True)
class OHLCVBar(MarketDataPoint):
    """OHLCV bar representing aggregated price information."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    interval_seconds: int
    kind: DataKind = field(default=DataKind.OHLCV, init=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        open_price = _to_decimal(self.open)
        high_price = _to_decimal(self.high)
        low_price = _to_decimal(self.low)
        close_price = _to_decimal(self.close)
        volume = _to_decimal(self.volume)
        if any(value < 0 for value in (open_price, high_price, low_price, close_price, volume)):
            raise ValueError("OHLCV values must be non-negative")
        if high_price < low_price:
            raise ValueError("high price must be greater or equal to low price")
        if not (low_price <= open_price <= high_price):
            raise ValueError("open price must lie between low and high")
        if not (low_price <= close_price <= high_price):
            raise ValueError("close price must lie between low and high")
        if self.interval_seconds < 1:
            raise ValueError("interval_seconds must be positive")
        object.__setattr__(self, "open", open_price)
        object.__setattr__(self, "high", high_price)
        object.__setattr__(self, "low", low_price)
        object.__setattr__(self, "close", close_price)
        object.__setattr__(self, "volume", volume)


@dataclass(frozen=True)
class AggregateMetric(MarketDataPoint):
    """Generic aggregated value produced from raw market data."""

    metric: str
    value: Decimal
    window_seconds: int
    kind: DataKind = field(default=DataKind.AGGREGATE, init=False)

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if not self.metric:
            raise ValueError("metric must be a non-empty string")
        value = _to_decimal(self.value)
        if self.window_seconds < 1:
            raise ValueError("window_seconds must be positive")
        object.__setattr__(self, "value", value)


# Backwards compatibility export ---------------------------------------------------------

Ticker = PriceTick

