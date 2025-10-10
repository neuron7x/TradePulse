# SPDX-License-Identifier: MIT
"""Core data utilities and models for TradePulse."""

import numpy as _np

if not hasattr(_np, "string_"):
    _np.string_ = _np.bytes_
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64

from .models import (
    AggregateMetric,
    DataKind,
    InstrumentType,
    MarketDataPoint,
    MarketMetadata,
    OHLCVBar,
    PriceTick,
    Ticker,
)
from .validation import (
    TimeSeriesValidationConfig,
    TimeSeriesValidationError,
    ValueColumnConfig,
    build_timeseries_schema,
    validate_timeseries_frame,
)
from .timeutils import (
    MarketCalendar,
    MarketCalendarRegistry,
    convert_timestamp,
    get_market_calendar,
    is_market_open,
    normalize_timestamp,
    to_utc,
)

__all__ = [
    "AggregateMetric",
    "DataKind",
    "InstrumentType",
    "MarketCalendar",
    "MarketCalendarRegistry",
    "MarketDataPoint",
    "MarketMetadata",
    "OHLCVBar",
    "PriceTick",
    "Ticker",
    "TimeSeriesValidationConfig",
    "TimeSeriesValidationError",
    "ValueColumnConfig",
    "build_timeseries_schema",
    "validate_timeseries_frame",
    "convert_timestamp",
    "get_market_calendar",
    "is_market_open",
    "normalize_timestamp",
    "to_utc",
]
