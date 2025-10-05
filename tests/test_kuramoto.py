# SPDX-License-Identifier: MIT
import numpy as np
from core.indicators.kuramoto import compute_phase, kuramoto_order

def test_kuramoto_range():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    ph = compute_phase(x)
    R = kuramoto_order(ph[-200:])
    assert 0.0 <= R <= 1.0
