# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.data.preprocess import normalize_df, scale_series


def test_normalize_df_orders_and_interpolates_numeric_columns() -> None:
    df = pd.DataFrame(
        {
            "ts": [3, 1, 2, 1],
            "value": [np.nan, 2.0, np.nan, 2.0],
            "label": ["c", "b", "a", "b"],
        }
    )

    normalized = normalize_df(df)

    expected_ts = pd.Series(
        pd.to_datetime([1, 2, 3], unit="s", utc=True), name="ts"
    )
    expected_values = pd.Series([2.0, 2.0, 2.0], name="value")

    pd.testing.assert_series_equal(normalized["ts"], expected_ts, check_names=False)
    pd.testing.assert_series_equal(normalized["value"], expected_values)
    assert list(normalized["label"]) == ["b", "a", "c"]


def test_normalize_df_without_timestamp_column() -> None:
    df = pd.DataFrame(
        {
            "value": [1.0, 1.0, np.nan],
            "category": ["x", "x", "y"],
        }
    )

    normalized = normalize_df(df, timestamp_col="nonexistent")

    pd.testing.assert_index_equal(normalized.index, pd.RangeIndex(0, 2))
    pd.testing.assert_series_equal(
        normalized["value"], pd.Series([1.0, 1.0], name="value")
    )
    assert list(normalized["category"]) == ["x", "y"]


def test_scale_series_zscore_and_minmax() -> None:
    data = [1.0, 2.0, 3.0]

    z_scaled = scale_series(data, method="zscore")
    minmax_scaled = scale_series(data, method="minmax")

    np.testing.assert_allclose(z_scaled.mean(), 0.0)
    np.testing.assert_allclose(z_scaled.std(), 1.0)
    np.testing.assert_allclose(minmax_scaled, np.array([0.0, 0.5, 1.0]))


def test_scale_series_handles_degenerate_inputs() -> None:
    zeros = np.zeros(5)
    np.testing.assert_allclose(scale_series(zeros), zeros)
    np.testing.assert_allclose(scale_series(zeros, method="minmax"), zeros)


def test_scale_series_validates_inputs() -> None:
    with pytest.raises(ValueError):
        scale_series(np.ones((2, 2)))

    with pytest.raises(ValueError):
        scale_series([1, 2, 3], method="invalid")

    with pytest.raises(TypeError):
        scale_series("123")
