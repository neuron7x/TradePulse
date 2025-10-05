# SPDX-License-Identifier: MIT
import numpy as np
from core.indicators.entropy import entropy, delta_entropy

def test_entropy_monotonic_bins():
    x = np.random.randn(1000)
    assert entropy(x, bins=10) >= 0
    assert entropy(x, bins=50) >= 0

def test_delta_entropy_window_short_ok():
    x = np.random.randn(50)
    assert isinstance(delta_entropy(x, window=100), float)
