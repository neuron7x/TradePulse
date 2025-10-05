# SPDX-License-Identifier: MIT
import numpy as np

from core.indicators.hurst import HurstExponentFeature, hurst_exponent


def test_hurst_bounds():
    x = np.cumsum(np.random.randn(1000))
    H = hurst_exponent(x)
    assert 0.0 <= H <= 1.0


def test_hurst_feature_metadata_and_value():
    x = np.cumsum(np.random.randn(1200))
    feature = HurstExponentFeature(min_lag=4, max_lag=40)
    value = feature.transform(x)
    assert 0.0 <= value <= 1.0
    assert feature.metadata()["params"] == {"min_lag": 4, "max_lag": 40}
    assert np.isclose(value, hurst_exponent(x, min_lag=4, max_lag=40))
