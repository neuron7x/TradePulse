# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from core.indicators.entropy import delta_entropy, entropy


def test_entropy_uniform_distribution_matches_log_bins(uniform_series: np.ndarray) -> None:
    bins = 20
    result = entropy(uniform_series, bins=bins)
    expected = np.log(bins)
    assert abs(result - expected) < 0.15, f"Entropy {result} deviates from log(bins) {expected}"


def test_entropy_degenerate_distribution_near_zero() -> None:
    series = np.ones(128)
    result = entropy(series, bins=10)
    assert result < 1e-9


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
