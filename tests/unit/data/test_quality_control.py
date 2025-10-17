from __future__ import annotations

import pandas as pd
import pytest

from core.data.quality_control import (
    QualityGateConfig,
    QualityGateError,
    QualityReport,
    RangeCheck,
    TemporalContract,
    quarantine_anomalies,
    validate_and_quarantine,
)
from core.data.validation import TimeSeriesValidationConfig, ValueColumnConfig


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": index,
            "close": [100, 101, 150, 102, 103, 104],
        }
    )
    data.loc[5, "timestamp"] = data.loc[4, "timestamp"]
    return data


def test_quarantine_detects_spikes_and_duplicates():
    frame = _frame().set_index("timestamp")
    result = quarantine_anomalies(frame, threshold=2.0, window=2, price_column="close")
    assert not result["duplicates"].empty
    assert not result["spikes"].empty
    assert result["clean"].shape[0] < frame.shape[0]


def test_validate_and_quarantine_integrates_schema():
    frame = _frame()
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=2.0,
        anomaly_window=2,
        max_quarantine_fraction=1.0,
    )
    report = validate_and_quarantine(frame, gate)
    assert isinstance(report, QualityReport)
    assert set(report.clean.columns) >= {"timestamp", "close"}
    assert not report.quarantined.empty
    assert report.blocked is False


def test_range_gate_blocks_out_of_bounds_rows():
    frame = _frame()
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        range_checks=(RangeCheck(column="close", max_value=120.0),),
    )
    report = validate_and_quarantine(frame, gate)
    assert report.blocked is True
    assert "close" in report.range_violations
    with pytest.raises(QualityGateError):
        report.raise_if_blocked()


def test_temporal_contract_flags_stale_batches():
    frame = _frame()
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        temporal_contract=TemporalContract(max_lag="0s"),
    )
    report = validate_and_quarantine(frame, gate)
    assert report.blocked is True
    assert report.contract_breaches
    with pytest.raises(QualityGateError):
        report.raise_if_blocked()

