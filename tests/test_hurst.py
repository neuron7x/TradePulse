# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.hurst import HurstFeature, hurst_exponent


def test_hurst_bounds():
    x = np.cumsum(np.random.randn(1000))
    H = hurst_exponent(x)
    assert 0.0 <= H <= 1.0


def test_hurst_returns_half_for_short_series():
    short_series = np.arange(5, dtype=float)
    H = hurst_exponent(short_series, max_lag=10)
    assert H == 0.5


def test_hurst_feature_metadata():
    x = np.cumsum(np.random.randn(1000))
    feature = HurstFeature(min_lag=5, max_lag=30)
    result = feature(x)
    assert result.metadata == {"min_lag": 5, "max_lag": 30}


def test_hurst_feature_custom_name():
    x = np.cumsum(np.random.randn(512))
    feature = HurstFeature(name="hurst_signal")
    result = feature(x)
    assert result.name == "hurst_signal"
