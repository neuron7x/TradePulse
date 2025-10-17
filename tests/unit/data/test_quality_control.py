from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import core.data.quality_control as qc
from core.data.quality_control import (
    QualityGateConfig,
    QualityGateError,
    QualityReport,
    RangeCheck,
    TemporalContract,
    quarantine_anomalies,
    validate_and_quarantine,
)
from pydantic import ValidationError
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


def test_range_check_requires_bounds():
    with pytest.raises(ValidationError) as exc:
        RangeCheck(column="close")
    assert "must define at least a minimum or maximum" in str(exc.value)


def test_range_check_rejects_inverted_bounds():
    with pytest.raises(ValidationError) as exc:
        RangeCheck(column="close", min_value=10.0, max_value=5.0)
    assert "minimum bound exceeds maximum" in str(exc.value)


def test_temporal_contract_rejects_conflicting_bounds():
    earliest = pd.Timestamp("2024-01-01T00:00:00Z")
    latest = pd.Timestamp("2024-01-01T03:00:00Z")
    with pytest.raises(ValidationError) as early_exc:
        TemporalContract(earliest=earliest, expected_start=earliest - pd.Timedelta(minutes=1))
    assert "expected_start" in str(early_exc.value)
    with pytest.raises(ValidationError) as late_exc:
        TemporalContract(latest=latest, expected_end=latest + pd.Timedelta(minutes=1))
    assert "expected_end" in str(late_exc.value)


def test_apply_range_checks_flags_missing_columns():
    frame = _frame()
    with pytest.raises(QualityGateError):
        qc._apply_range_checks(frame, (RangeCheck(column="volume", min_value=0.0),), "timestamp")


def test_apply_range_checks_respect_inclusive_flags():
    index = pd.date_range("2024-02-01", periods=5, freq="1min", tz="UTC")
    payload = pd.DataFrame({"timestamp": index, "value": [0.0, 1.0, np.nan, 10.0, 20.0]})
    min_result = qc._apply_range_checks(
        payload,
        (RangeCheck(column="value", min_value=1.0, inclusive_min=True),),
        "timestamp",
    )
    assert list(min_result["value"].index) == [index[0]]
    max_result = qc._apply_range_checks(
        payload,
        (RangeCheck(column="value", max_value=10.0, inclusive_max=False),),
        "timestamp",
    )
    assert list(max_result["value"].index) == [index[3], index[4]]


def test_quality_gate_config_validates_inputs():
    schema = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    with pytest.raises(ValidationError) as price_exc:
        QualityGateConfig(schema=schema, price_column="open")
    assert "price_column" in str(price_exc.value)
    with pytest.raises(ValidationError) as dup_exc:
        QualityGateConfig(
            schema=schema,
            price_column="close",
            range_checks=(
                RangeCheck(column="close", min_value=90.0),
                RangeCheck(column="close", max_value=110.0),
            ),
        )
    assert "Duplicate range checks" in str(dup_exc.value)


def test_enforce_temporal_contract_reports_all_breaches(monkeypatch):
    timestamps = pd.Series(pd.date_range("2024-03-01", periods=3, freq="1h", tz="UTC"))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, tz=None: timestamps.iloc[-1] + pd.Timedelta(hours=1)),
    )
    contract = TemporalContract(
        earliest="2024-03-01T00:30:00",
        latest=pd.Timestamp("2024-03-01T01:30:00Z"),
        expected_start="2024-03-01T01:10:00",
        expected_end=pd.Timestamp("2024-02-28T20:00:00", tz="America/New_York"),
        tolerance=pd.Timedelta(0),
        max_lag=timedelta(seconds=0),
    )
    breaches, blocked = qc._enforce_temporal_contract(timestamps, contract)
    assert blocked is True
    assert any("earliest" in message for message in breaches)
    assert any("latest" in message for message in breaches)
    assert any("starts" in message for message in breaches)
    assert any("ends" in message for message in breaches)
    assert any("stale" in message for message in breaches)


def test_enforce_temporal_contract_rejects_timezone_mismatch():
    timestamps = pd.Series(pd.date_range("2024-04-01", periods=2, freq="1h"))
    contract = TemporalContract(earliest=pd.Timestamp("2024-04-01T00:00:00Z"))
    with pytest.raises(QualityGateError):
        qc._enforce_temporal_contract(timestamps, contract)


def test_validate_and_quarantine_blocks_on_quarantine_ratio():
    frame = _frame()
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=1.0,
        anomaly_window=2,
        max_quarantine_fraction=0.0,
    )
    report = validate_and_quarantine(frame, gate)
    assert report.blocked is True
    with pytest.raises(QualityGateError) as exc:
        report.raise_if_blocked()
    assert "Quarantine ratio" in str(exc.value)


def test_quality_report_raise_if_not_blocked_returns():
    report = QualityReport(
        clean=pd.DataFrame(),
        quarantined=pd.DataFrame(),
        duplicates=pd.DataFrame(),
        spikes=pd.DataFrame(),
    )
    report.raise_if_blocked()


def test_quarantine_anomalies_empty_frame():
    frame = pd.DataFrame(columns=["close"])
    result = quarantine_anomalies(frame, threshold=3.0, window=3, price_column="close")
    assert result["clean"].empty and result["spikes"].empty and result["duplicates"].empty


def test_apply_range_checks_without_violations_returns_empty():
    index = pd.date_range("2024-02-02", periods=3, freq="1min", tz="UTC")
    payload = pd.DataFrame({"timestamp": index, "value": [5.0, 5.5, 5.2]})
    result = qc._apply_range_checks(
        payload,
        (RangeCheck(column="value", min_value=1.0, max_value=10.0),),
        "timestamp",
    )
    assert result == {}


def test_enforce_temporal_contract_returns_on_empty_series():
    timestamps = pd.Series(dtype="datetime64[ns]")
    breaches, blocked = qc._enforce_temporal_contract(timestamps, TemporalContract())
    assert breaches == ()
    assert blocked is False


def test_enforce_temporal_contract_respects_tolerance(monkeypatch):
    timestamps = pd.Series(pd.date_range("2024-05-01", periods=3, freq="1h", tz="UTC"))
    monkeypatch.setattr(
        pd.Timestamp,
        "now",
        classmethod(lambda cls, tz=None: timestamps.iloc[-1] + pd.Timedelta(minutes=30)),
    )
    contract = TemporalContract(
        earliest="2024-05-01T00:00:00",
        latest="2024-05-01T02:00:00Z",
        expected_start="2024-05-01T00:00:30",
        expected_end="2024-05-01T01:59:30Z",
        tolerance=timedelta(minutes=1),
        max_lag="2h",
    )
    breaches, blocked = qc._enforce_temporal_contract(timestamps, contract)
    assert breaches == ()
    assert blocked is False


def test_enforce_temporal_contract_handles_naive_inputs():
    timestamps = pd.Series(pd.date_range("2024-06-01", periods=3, freq="1h"))
    contract = TemporalContract(
        earliest="2024-06-01T00:00:00",
        latest="2024-06-01T02:00:00",
        tolerance=None,
        max_lag=None,
    )
    breaches, blocked = qc._enforce_temporal_contract(timestamps, contract)
    assert breaches == ()
    assert blocked is False


def test_validate_and_quarantine_range_violation_without_preexisting_quarantine():
    index = pd.date_range("2024-07-01", periods=3, freq="1min", tz="UTC")
    frame = pd.DataFrame({"timestamp": index, "close": [100.0, 101.0, 150.0]})
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=1000.0,
        anomaly_window=3,
        range_checks=(RangeCheck(column="close", max_value=120.0),),
        max_quarantine_fraction=1.0,
    )
    report = validate_and_quarantine(frame, gate)
    assert not report.clean["timestamp"].eq(index[-1]).any()
    assert report.spikes.empty
    assert report.quarantined.loc[report.quarantined["timestamp"] == index[-1]].shape[0] == 1

