from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from pydantic import ValidationError

from core.data.quality_control import (
    QualityGateConfig,
    QualityGateError,
    QualityReport,
    RangeCheck,
    TemporalContract,
    _apply_range_checks,
    _enforce_temporal_contract,
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


def test_range_check_requires_bounds() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RangeCheck(column="close")
    assert "RangeCheck must define" in str(excinfo.value)


def test_range_check_lower_bound_cannot_exceed_upper() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RangeCheck(column="close", min_value=2.0, max_value=1.0)
    assert "minimum bound exceeds" in str(excinfo.value)


def test_quality_gate_config_rejects_unknown_price_column() -> None:
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    with pytest.raises(ValidationError) as excinfo:
        QualityGateConfig(schema=config, price_column="open")
    assert "price_column" in str(excinfo.value)


def test_quality_gate_config_detects_duplicate_range_checks() -> None:
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    with pytest.raises(ValidationError) as excinfo:
        QualityGateConfig(
            schema=config,
            price_column="close",
            range_checks=(
                RangeCheck(column="close", min_value=0.0),
                RangeCheck(column="close", max_value=100.0),
            ),
        )
    assert "Duplicate range checks" in str(excinfo.value)


def test_quality_report_raise_if_blocked_collects_details() -> None:
    now = pd.Timestamp("2024-01-01T00:00:00Z")
    payload = pd.DataFrame({"timestamp": [now], "close": [200.0]}).set_index(
        "timestamp"
    )
    report = QualityReport(
        clean=pd.DataFrame(),
        quarantined=pd.DataFrame(),
        duplicates=pd.DataFrame(),
        spikes=pd.DataFrame(),
        range_violations={"close": payload},
        contract_breaches=("Breached",),
        blocked=True,
    )
    with pytest.raises(QualityGateError) as excinfo:
        report.raise_if_blocked()
    message = str(excinfo.value)
    assert "Breached" in message
    assert "close" in message


def test_quality_report_blocked_without_reason_defaults_to_quarantine() -> None:
    report = QualityReport(
        clean=pd.DataFrame(),
        quarantined=pd.DataFrame({"timestamp": [pd.Timestamp.utcnow()]}),
        duplicates=pd.DataFrame(),
        spikes=pd.DataFrame(),
        blocked=True,
    )
    with pytest.raises(QualityGateError) as excinfo:
        report.raise_if_blocked()
    assert "Quarantine ratio" in str(excinfo.value)


def test_quarantine_anomalies_handles_empty_frame() -> None:
    frame = pd.DataFrame(columns=["close"])
    result = quarantine_anomalies(frame, threshold=1.0, window=3, price_column="close")
    assert all(bucket.empty for bucket in result.values())


def test_apply_range_checks_respects_inclusive_flags() -> None:
    index = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"])
    frame = pd.DataFrame(
        {"timestamp": index, "close": [100.0, 105.0], "volume": [10, float("nan")]} 
    )
    checks = (
        RangeCheck(column="close", min_value=100.0, inclusive_min=False),
        RangeCheck(column="volume", max_value=15.0, inclusive_max=False),
    )
    violations = _apply_range_checks(frame, checks, "timestamp")
    assert set(violations) == {"close"}
    assert violations["close"].index[0] == index[0]


def test_apply_range_checks_missing_column_is_error() -> None:
    frame = _frame().set_index("timestamp")
    with pytest.raises(QualityGateError):
        _apply_range_checks(frame, (RangeCheck(column="volume", min_value=0.0),), "timestamp")


def test_enforce_temporal_contract_detects_mismatched_timezones() -> None:
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=2, freq="1min"))
    contract = TemporalContract(earliest=pd.Timestamp("2024-01-01T00:00:00Z"))
    with pytest.raises(QualityGateError):
        _enforce_temporal_contract(timestamps, contract)


def test_enforce_temporal_contract_reports_breaches_and_lag(monkeypatch: pytest.MonkeyPatch) -> None:
    tz_index = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1min", tz="UTC")
    timestamps = pd.Series(tz_index)
    contract = TemporalContract(
        earliest=tz_index[1],
        latest=tz_index[1],
        expected_start=tz_index[2],
        expected_end=tz_index[0],
        tolerance=pd.Timedelta(seconds=10),
        max_lag=pd.Timedelta(seconds=1),
    )

    fixed_now = tz_index[-1] + pd.Timedelta(minutes=5)
    monkeypatch.setattr(pd.Timestamp, "now", lambda tz=None: fixed_now)

    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert blocked is True
    assert len(breaches) >= 3
    assert any("First timestamp" in item for item in breaches)
    assert any("Last timestamp" in item for item in breaches)
    assert any("Batch starts" in item for item in breaches)
    assert any("Batch ends" in item for item in breaches)
    assert any("stale" in item for item in breaches)


def test_enforce_temporal_contract_aligns_naive_contract_to_series_timezone() -> None:
    tz_index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min", tz="UTC")
    timestamps = pd.Series(tz_index)
    contract = TemporalContract(expected_end=pd.Timestamp("2024-01-01T00:01:00"))
    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert not breaches
    assert blocked is False


def test_temporal_contract_accepts_string_inputs() -> None:
    contract = TemporalContract(
        earliest="2024-01-01T00:00:00Z",
        latest="2024-01-01T01:00:00Z",
        expected_start="2024-01-01T00:00:00Z",
        expected_end="2024-01-01T01:00:00Z",
        tolerance="5s",
        max_lag="30s",
    )
    assert isinstance(contract.earliest, pd.Timestamp)
    assert contract.tolerance == pd.Timedelta(seconds=5)
    assert contract.max_lag == pd.Timedelta(seconds=30)


def test_temporal_contract_rejects_expected_start_before_earliest() -> None:
    with pytest.raises(ValidationError):
        TemporalContract(earliest="2024-01-01T00:01:00Z", expected_start="2024-01-01T00:00:00Z")


def test_temporal_contract_defaults_optional_durations() -> None:
    contract = TemporalContract(tolerance=None, max_lag=None)
    assert contract.tolerance == pd.Timedelta(0)
    assert contract.max_lag is None


def test_temporal_contract_rejects_expected_end_after_latest() -> None:
    with pytest.raises(ValidationError):
        TemporalContract(latest="2024-01-01T00:01:00Z", expected_end="2024-01-01T00:02:00Z")


def test_temporal_contract_accepts_python_timedelta_objects() -> None:
    contract = TemporalContract(tolerance=timedelta(seconds=3), max_lag=timedelta(seconds=7))
    assert contract.tolerance == pd.Timedelta(seconds=3)
    assert contract.max_lag == pd.Timedelta(seconds=7)


def test_temporal_contract_handles_timezone_conversion() -> None:
    tz_index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min", tz="UTC")
    timestamps = pd.Series(tz_index)
    contract = TemporalContract(earliest=pd.Timestamp("2023-12-31T19:00:00", tz="US/Eastern"))
    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert not breaches
    assert blocked is False


def test_quality_report_default_block_message_when_no_context() -> None:
    report = QualityReport(
        clean=pd.DataFrame(),
        quarantined=pd.DataFrame(),
        duplicates=pd.DataFrame(),
        spikes=pd.DataFrame(),
        blocked=True,
    )
    with pytest.raises(QualityGateError) as excinfo:
        report.raise_if_blocked()
    assert "Batch blocked by quality gate" in str(excinfo.value)


def test_validate_and_quarantine_blocks_when_quarantine_ratio_exceeded() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "close": [100.0, 500.0, 510.0, 520.0],
        }
    )
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
    with pytest.raises(QualityGateError):
        report.raise_if_blocked()


def test_validate_and_quarantine_moves_range_violations_into_quarantine() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    frame = pd.DataFrame({"timestamp": index, "close": [100.0, 200.0]})
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=10.0,
        anomaly_window=2,
        range_checks=(RangeCheck(column="close", max_value=150.0),),
        max_quarantine_fraction=1.0,
    )
    report = validate_and_quarantine(frame, gate)
    assert report.blocked is True
    assert report.clean.shape[0] == 1
    assert not report.quarantined.empty


def test_apply_range_checks_with_inclusive_bounds() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    frame = pd.DataFrame({"timestamp": index, "close": [100.0, 120.0]})
    violations = _apply_range_checks(
        frame,
        (
            RangeCheck(column="close", min_value=100.0),
            RangeCheck(column="close", max_value=120.0),
        ),
        "timestamp",
    )
    assert violations == {}


def test_apply_range_checks_ignores_nan_values() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    frame = pd.DataFrame({"timestamp": index, "close": [float("nan"), float("nan")]})
    violations = _apply_range_checks(
        frame,
        (RangeCheck(column="close", min_value=0.0, inclusive_min=False),),
        "timestamp",
    )
    assert violations == {}


def test_quality_report_raise_if_blocked_noop_when_not_blocked() -> None:
    report = QualityReport(
        clean=pd.DataFrame(),
        quarantined=pd.DataFrame(),
        duplicates=pd.DataFrame(),
        spikes=pd.DataFrame(),
        blocked=False,
    )
    report.raise_if_blocked()


def test_validate_and_quarantine_merges_range_violations_with_existing_quarantine() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp": [index[0], index[0], index[1]],
            "close": [100.0, 100.0, 250.0],
        }
    )
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=10.0,
        anomaly_window=2,
        range_checks=(RangeCheck(column="close", max_value=200.0),),
        max_quarantine_fraction=1.0,
    )
    report = validate_and_quarantine(frame, gate)
    quarantined_index = set(report.quarantined["timestamp"])
    assert index[0] in quarantined_index  # duplicate path retained
    assert index[1] in quarantined_index  # range violation merged
    assert not report.clean["timestamp"].eq(index[2]).any()


def test_validate_and_quarantine_handles_all_rows_becoming_range_violations() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min")
    frame = pd.DataFrame({"timestamp": index, "close": [250.0, 260.0]})
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    gate = QualityGateConfig(
        schema=config,
        price_column="close",
        anomaly_threshold=10.0,
        anomaly_window=2,
        range_checks=(RangeCheck(column="close", max_value=200.0),),
        max_quarantine_fraction=1.0,
    )
    report = validate_and_quarantine(frame, gate)
    assert report.clean.empty
    assert not report.quarantined.empty
    assert report.quarantined.shape[0] == frame.shape[0]


def test_enforce_temporal_contract_handles_empty_series() -> None:
    timestamps = pd.Series(dtype="datetime64[ns]")
    contract = TemporalContract()
    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert breaches == ()
    assert blocked is False


def test_enforce_temporal_contract_allows_naive_timestamps() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1min")
    timestamps = pd.Series(index)
    contract = TemporalContract(earliest=pd.Timestamp("2024-01-01 00:00:00"))
    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert breaches == ()
    assert blocked is False


def test_enforce_temporal_contract_detects_expected_start_tolerance_violation(monkeypatch) -> None:
    tz_index = pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="1min", tz="UTC")
    timestamps = pd.Series(tz_index)
    contract = TemporalContract(
        expected_start=tz_index[1],
        expected_end=tz_index[0],
        tolerance=pd.Timedelta(seconds=5),
        max_lag=pd.Timedelta(seconds=1),
    )
    fixed_now = tz_index[-1] + pd.Timedelta(seconds=2)
    monkeypatch.setattr(pd.Timestamp, "now", lambda tz=None: fixed_now)
    breaches, blocked = _enforce_temporal_contract(timestamps, contract)
    assert blocked is True
    assert any("Batch starts" in message for message in breaches)
