# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from core.indicators.hurst import hurst_exponent


def test_hurst_exponent_of_brownian_motion_near_half(brownian_motion: np.ndarray) -> None:
    H = hurst_exponent(brownian_motion, min_lag=2, max_lag=40)
    assert 0.45 <= H <= 0.55, f"Hurst exponent {H} deviates from Brownian expectation"


def test_hurst_returns_default_for_short_series() -> None:
    series = np.linspace(0, 1, 10)
    assert hurst_exponent(series, max_lag=20) == 0.5


def test_hurst_clips_to_unit_interval(brownian_motion: np.ndarray) -> None:
    H = hurst_exponent(brownian_motion * 10, min_lag=2, max_lag=80)
    assert 0.0 <= H <= 1.0
