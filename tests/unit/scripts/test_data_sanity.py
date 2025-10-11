from __future__ import annotations

import pandas as pd
import pytest

from scripts import data_sanity


def _write_sample_csv(path: str | bytes):
    df = pd.DataFrame(
        {
            "ts": [
                "2021-01-01T00:00:00Z",
                "2021-01-01T00:00:10Z",
                "2021-01-01T00:00:25Z",
                "2021-01-01T00:00:25Z",
            ],
            "value": [1, 2, None, None],
            "category": ["a", "a", "b", "b"],
            "status": ["ok", "ok", "ok", "ok"],
        }
    )
    df.to_csv(path, index=False)
    return df


def test_analyze_csv_returns_expected_metrics(tmp_path):
    csv_path = tmp_path / "example.csv"
    df = _write_sample_csv(csv_path)

    analysis = data_sanity.analyze_csv(csv_path)

    assert analysis.path == csv_path
    assert analysis.row_count == len(df)
    assert analysis.column_count == len(df.columns)
    assert analysis.duplicate_rows == 1
    assert analysis.nan_ratio == pytest.approx(df.isna().mean().mean())
    assert analysis.row_nan_ratio == pytest.approx(df.isna().any(axis=1).mean())
    assert analysis.column_nan_ratios["value"] == pytest.approx(0.5)
    assert analysis.constant_columns == ["status"]
    assert analysis.dtypes["value"] == str(df.dtypes["value"])
    assert analysis.timestamp_gap_stats is not None
    assert analysis.timestamp_gap_stats.median_seconds == pytest.approx(10.0)
    assert analysis.timestamp_gap_stats.max_seconds == pytest.approx(15.0)
    assert analysis.timestamp_gap_stats.min_seconds == pytest.approx(0.0)
    assert analysis.timestamp_gap_stats.negative_gap_count == 0


def test_format_analysis_includes_column_limits(tmp_path):
    csv_path = tmp_path / "example.csv"
    _write_sample_csv(csv_path)

    analysis = data_sanity.analyze_csv(csv_path)
    report = data_sanity.format_analysis(
        analysis, max_column_details=1, max_constant_columns=1
    )

    assert "column NaN ratios" in report
    assert "value=50.00%" in report
    assert "…" not in report
    assert "constant columns" in report
    assert "status" in report


def test_main_handles_cli_execution(tmp_path, capsys):
    csv_path = tmp_path / "dataset.csv"
    _write_sample_csv(csv_path)

    exit_code = data_sanity.main([str(tmp_path), "--max-constant-columns", "2"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"# File: {csv_path}" in output
    assert "duplicates" in output
    assert "constant columns" in output
