# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.hurst import HurstFeature, hurst_exponent


def test_hurst_exponent_of_brownian_motion_near_half(brownian_motion: np.ndarray) -> None:
    H = hurst_exponent(brownian_motion, min_lag=2, max_lag=40)
    assert 0.45 <= H <= 0.55, f"Hurst exponent {H} deviates from Brownian expectation"


def test_hurst_returns_default_for_short_series() -> None:
    series = np.linspace(0, 1, 10)
    assert hurst_exponent(series, max_lag=20) == 0.5


def test_hurst_clips_to_unit_interval(brownian_motion: np.ndarray) -> None:
    H = hurst_exponent(brownian_motion * 10, min_lag=2, max_lag=80)
    assert 0.0 <= H <= 1.0


def test_hurst_feature_returns_metadata(brownian_motion: np.ndarray) -> None:
    feature = HurstFeature(min_lag=3, max_lag=30, name="hurst")
    outcome = feature.transform(brownian_motion)
    assert outcome.name == "hurst"
    assert outcome.metadata == {"min_lag": 3, "max_lag": 30}
    expected = hurst_exponent(brownian_motion, min_lag=3, max_lag=30)
    assert outcome.value == pytest.approx(expected, rel=1e-12)
