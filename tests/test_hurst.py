# SPDX-License-Identifier: MIT
import numpy as np
from core.indicators.hurst import hurst_exponent

def test_hurst_bounds():
    x = np.cumsum(np.random.randn(1000))
    H = hurst_exponent(x)
    assert 0.0 <= H <= 1.0
