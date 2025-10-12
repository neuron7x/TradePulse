from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from analytics.finops import build_daily_cost_report


def test_build_daily_cost_report_generates_confidence_and_regression() -> None:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "team": ["alpha"] * 12,
            "environment": ["prod"] * 12,
            "service": ["research"] * 12,
            "cpu_cost": [100 + i * 15 for i in range(12)],
            "gpu_cost": [40 + i * 5 for i in range(12)],
            "io_cost": [10 + i * 2 for i in range(12)],
        }
    )

    report = build_daily_cost_report(
        frame,
        segments=["team", "environment"],
        confidence=0.9,
        baseline_window=5,
        trend_threshold=0.05,
        zscore_threshold=1.0,
    )

    assert not report.daily_totals.empty
    assert {"segment", "cpu_cost", "gpu_cost", "io_cost", "total_cost"}.issubset(report.daily_totals.columns)

    latest_confidence = (
        report.confidence_intervals.sort_values(["segment", "resource", "date"]).groupby(["segment", "resource"]).tail(1)
    )
    assert not latest_confidence.empty
    assert latest_confidence["upper"].ge(latest_confidence["lower"]).all()

    assert report.regression_signals, "expected regression signals for upward trend"

    markdown = report.to_markdown()
    assert "Daily Cost Report" in markdown
    assert "Confidence Bands" in markdown


def test_build_daily_cost_report_requires_timestamp_column() -> None:
    frame = pd.DataFrame({"cpu_cost": [1, 2, 3]})
    with pytest.raises(ValueError):
        build_daily_cost_report(frame)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"confidence": 1.0}, "confidence"),
        ({"baseline_window": 1}, "baseline_window"),
        ({"trend_threshold": 0.0}, "trend_threshold"),
        ({"zscore_threshold": 0.0}, "zscore_threshold"),
    ],
)
def test_build_daily_cost_report_validates_parameters(kwargs: dict[str, Any], message: str) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "cpu_cost": [10.0, 11.0, 12.0],
            "gpu_cost": [5.0, 5.5, 6.0],
            "io_cost": [2.0, 2.5, 3.0],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        build_daily_cost_report(frame, **kwargs)
    err_message = str(excinfo.value)
    # Ensure the parameter name is surfaced so operators can quickly spot the
    # offending control knob.
    assert message in err_message
    # The validation errors should also echo the rejected value to simplify
    # debugging misconfigured FinOps runs.
    failing_value = next(iter(kwargs.values()))
    assert repr(failing_value) in err_message
