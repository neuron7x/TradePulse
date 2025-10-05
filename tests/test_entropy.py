# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.base import FeatureBlock
from core.indicators.entropy import (
    DeltaEntropyFeature,
    EntropyFeature,
    delta_entropy,
    entropy,
)


def test_entropy_monotonic_bins():
    x = np.random.randn(1000)
    assert entropy(x, bins=10) >= 0
    assert entropy(x, bins=50) >= 0


def test_entropy_returns_zero_for_empty_input():
    assert entropy(np.array([])) == 0.0


def test_delta_entropy_window_short_ok():
    x = np.random.randn(50)
    assert delta_entropy(x, window=100) == 0.0


def test_delta_entropy_feature_metadata():
    x = np.random.randn(512)
    feature = DeltaEntropyFeature(window=64, bins_range=(8, 32))
    result = feature(x)
    assert result.metadata == {"window": 64, "bins_range": (8, 32)}


def test_entropy_feature_custom_name_and_metadata():
    x = np.random.randn(400)
    feature = EntropyFeature(bins=20, name="entropy_signal")
    result = feature(x)
    assert result.name == "entropy_signal"
    assert result.metadata == {"bins": 20}


def test_entropy_feature_block_interface():
    x = np.random.randn(400)
    block = FeatureBlock([EntropyFeature(bins=20), DeltaEntropyFeature(window=100)])
    result = block(x)
    assert set(result.keys()) == {"entropy", "delta_entropy"}
