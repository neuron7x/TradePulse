# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    kuramoto_order,
)

def test_kuramoto_range():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    ph = compute_phase(x)
    R = kuramoto_order(ph[-200:])
    assert 0.0 <= R <= 1.0


def test_kuramoto_feature_block_single_series():
    ph = np.random.uniform(-np.pi, np.pi, size=512)
    feature = KuramotoOrderFeature()
    result = feature(ph)
    assert 0.0 <= result.value <= 1.0


def test_multi_asset_kuramoto_feature_counts_assets():
    series = [np.sin(np.linspace(0, np.pi, 100)) for _ in range(3)]
    feature = MultiAssetKuramotoFeature()
    result = feature(series)
    assert result.metadata["assets"] == 3
