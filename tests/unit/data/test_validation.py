# SPDX-License-Identifier: MIT
"""Unit tests for the strict time series validation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.validation import (
    TimeSeriesValidationConfig,
    TimeSeriesValidationError,
    ValueColumnConfig,
    validate_timeseries_frame,
)


@pytest.fixture()
def base_config() -> TimeSeriesValidationConfig:
    return TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=(
            ValueColumnConfig(name="close", dtype="float64"),
            ValueColumnConfig(name="volume", dtype="float64"),
        ),
        frequency="1min",
        require_timezone="UTC",
    )


@pytest.fixture()
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01 00:00:00",
                periods=4,
                freq="1min",
                tz="UTC",
            ),
            "close": [100.0, 101.5, 103.2, 104.8],
            "volume": [10.0, 12.0, 11.5, 13.0],
        }
    )


def test_validate_timeseries_frame_success(base_config: TimeSeriesValidationConfig, valid_frame: pd.DataFrame) -> None:
    validated = validate_timeseries_frame(valid_frame, base_config)
    pd.testing.assert_frame_equal(validated, valid_frame)


def test_validate_timeseries_frame_rejects_nan(base_config: TimeSeriesValidationConfig, valid_frame: pd.DataFrame) -> None:
    frame = valid_frame.copy()
    frame.loc[2, "close"] = float("nan")

    with pytest.raises(TimeSeriesValidationError) as err:
        validate_timeseries_frame(frame, base_config)

    assert "NaN" in str(err.value)


def test_validate_timeseries_frame_rejects_duplicate_timestamps(
    base_config: TimeSeriesValidationConfig, valid_frame: pd.DataFrame
) -> None:
    frame = valid_frame.copy()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]

    with pytest.raises(TimeSeriesValidationError) as err:
        validate_timeseries_frame(frame, base_config)

    assert "duplicate" in str(err.value).lower()


def test_validate_timeseries_frame_enforces_frequency(
    base_config: TimeSeriesValidationConfig, valid_frame: pd.DataFrame
) -> None:
    frame = valid_frame.copy()
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"] + pd.Timedelta(minutes=2)
    frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"] + pd.Timedelta(minutes=1)

    with pytest.raises(TimeSeriesValidationError) as err:
        validate_timeseries_frame(frame, base_config)

    assert "frequency" in str(err.value)


def test_validate_timeseries_frame_detects_timezone_drift(
    base_config: TimeSeriesValidationConfig, valid_frame: pd.DataFrame
) -> None:
    frame = valid_frame.copy()
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("Europe/Berlin")

    with pytest.raises(TimeSeriesValidationError) as err:
        validate_timeseries_frame(frame, base_config)

    assert "utc" in str(err.value).lower()
