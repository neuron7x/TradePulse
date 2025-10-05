# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.kuramoto import kuramoto_order

def test_R_bounds_property():
    ph = np.random.uniform(-np.pi, np.pi, size=512)
    R = kuramoto_order(ph)
    assert 0.0 <= R <= 1.0
