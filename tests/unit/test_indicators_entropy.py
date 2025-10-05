# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.entropy import (
    DeltaEntropyFeature,
    EntropyFeature,
    delta_entropy,
    entropy,
)


def test_entropy_uniform_distribution_matches_log_bins(uniform_series: np.ndarray) -> None:
    bins = 20
    result = entropy(uniform_series, bins=bins)
    expected = np.log(bins)
    assert abs(result - expected) < 0.15, f"Entropy {result} deviates from log(bins) {expected}"


def test_entropy_degenerate_distribution_near_zero() -> None:
    series = np.ones(128)
    result = entropy(series, bins=10)
    assert result < 1e-9


def test_entropy_handles_extreme_values_and_non_finite() -> None:
    series = np.array(
        [
            0.0,
            0.0,
            1.0,
            np.finfo(float).max,
            -np.finfo(float).max / 10,
            np.nan,
            np.inf,
            -np.inf,
        ]
    )
    result = entropy(series, bins=16)
    assert np.isfinite(result)
    assert result >= 0.0


def test_entropy_of_empty_series_is_zero() -> None:
    assert entropy(np.array([])) == 0.0


def test_delta_entropy_requires_two_windows(peaked_series: np.ndarray) -> None:
    short_series = peaked_series[:100]
    assert delta_entropy(short_series, window=80) == 0.0


def test_delta_entropy_detects_spread_change() -> None:
    first = np.zeros(80)
    second = np.linspace(-1.0, 1.0, 80)
    series = np.concatenate([first, second])
    result = delta_entropy(series, window=80)
    assert result > 0.0, "Delta entropy should increase when distribution widens"


def test_entropy_feature_wraps_indicator(uniform_series: np.ndarray) -> None:
    feature = EntropyFeature(bins=15, name="custom_entropy")
    outcome = feature.transform(uniform_series)
    assert outcome.name == "custom_entropy"
    assert outcome.metadata == {"bins": 15}
    expected = entropy(uniform_series, bins=15)
    assert outcome.value == pytest.approx(expected, rel=1e-12)


def test_delta_entropy_feature_metadata(peaked_series: np.ndarray) -> None:
    feature = DeltaEntropyFeature(window=40, bins_range=(5, 25))
    outcome = feature.transform(peaked_series)
    assert outcome.name == "delta_entropy"
    assert outcome.metadata == {"window": 40, "bins_range": (5, 25)}
    expected = delta_entropy(peaked_series, window=40, bins_range=(5, 25))
    assert outcome.value == pytest.approx(expected, rel=1e-12)
