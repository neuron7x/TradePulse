from __future__ import annotations

import pandas as pd

from core.data.quality_control import QualityReport, quarantine_anomalies, validate_and_quarantine
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
    result = quarantine_anomalies(frame, threshold=2.0, window=2)
    assert not result["duplicates"].empty
    assert not result["spikes"].empty
    assert result["clean"].shape[0] < frame.shape[0]


def test_validate_and_quarantine_integrates_schema():
    frame = _frame()
    config = TimeSeriesValidationConfig(
        timestamp_column="timestamp",
        value_columns=[ValueColumnConfig(name="close", dtype="float64")],
    )
    report = validate_and_quarantine(frame, config, threshold=2.0, window=2)
    assert isinstance(report, QualityReport)
    assert set(report.clean.columns) >= {"timestamp", "close"}
    assert not report.quarantined.empty

