# SPDX-License-Identifier: MIT
import numpy as np
import pytest

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


def test_delta_entropy_window_short_ok():
    x = np.random.randn(50)
    assert isinstance(delta_entropy(x, window=100), float)


def test_entropy_feature_matches_function():
    x = np.random.randn(256)
    feature = EntropyFeature(bins=32)
    assert pytest.approx(feature.transform(x)) == entropy(x, bins=32)
    assert feature.metadata()["name"] == "entropy"
    assert feature.metadata()["params"]["bins"] == 32


def test_delta_entropy_feature_respects_bins_range():
    x = np.linspace(0, 1, 400)
    feature = DeltaEntropyFeature(window=100, bins_range=(12, 20))
    value = feature.transform(x)
    assert isinstance(value, float)
    assert pytest.approx(value) == delta_entropy(x, window=100, bins_range=(12, 20))
