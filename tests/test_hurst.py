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


def test_hurst_feature_metadata():
    x = np.cumsum(np.random.randn(1000))
    feature = HurstFeature(min_lag=5, max_lag=30)
    result = feature(x)
    assert result.metadata == {"min_lag": 5, "max_lag": 30}
