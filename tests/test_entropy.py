# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.base import FeatureBlock
from core.indicators.entropy import DeltaEntropyFeature, EntropyFeature, delta_entropy, entropy

def test_entropy_monotonic_bins():
    x = np.random.randn(1000)
    assert entropy(x, bins=10) >= 0
    assert entropy(x, bins=50) >= 0

def test_delta_entropy_window_short_ok():
    x = np.random.randn(50)
    assert isinstance(delta_entropy(x, window=100), float)


def test_entropy_feature_block_interface():
    x = np.random.randn(400)
    block = FeatureBlock([EntropyFeature(bins=20), DeltaEntropyFeature(window=100)])
    result = block(x)
    assert set(result.keys()) == {"entropy", "delta_entropy"}
