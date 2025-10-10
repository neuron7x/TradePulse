# SPDX-License-Identifier: MIT
"""Core data utilities and models for TradePulse."""

import numpy as _np

if not hasattr(_np, "string_"):
    _np.string_ = _np.bytes_
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64

from .catalog import normalize_symbol, normalize_venue
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
try:
    from .timeutils import (
        MarketCalendar,
        MarketCalendarRegistry,
        convert_timestamp,
        get_market_calendar,
        get_timezone,
        is_market_open,
        normalize_timestamp,
        to_utc,
        validate_bar_alignment,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    if exc.name != "exchange_calendars":
        raise

    def _missing_dependency(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "exchange_calendars is required for timeutils functionality in core.data"
        )

    MarketCalendar = MarketCalendarRegistry = object  # type: ignore
    convert_timestamp = get_market_calendar = get_timezone = is_market_open = _missing_dependency
    normalize_timestamp = to_utc = validate_bar_alignment = _missing_dependency

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
    "normalize_symbol",
    "normalize_venue",
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
    "get_timezone",
    "validate_bar_alignment",
]
