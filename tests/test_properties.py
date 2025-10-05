# SPDX-License-Identifier: MIT
import numpy as np
from core.indicators.kuramoto import kuramoto_order

def test_R_bounds_property():
    ph = np.random.uniform(-np.pi, np.pi, size=512)
    R = kuramoto_order(ph)
    assert 0.0 <= R <= 1.0
