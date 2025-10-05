# SPDX-License-Identifier: MIT
import numpy as np

from core.indicators.kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoBlock,
    PhaseFeature,
    compute_phase,
    kuramoto_order,
    multi_asset_kuramoto,
)


def test_kuramoto_range():
    x = np.sin(np.linspace(0, 10 * np.pi, 1000))
    ph = compute_phase(x)
    R = kuramoto_order(ph[-200:])
    assert 0.0 <= R <= 1.0


def test_phase_feature_metadata():
    feature = PhaseFeature(backend="fft")
    meta = feature.metadata()
    assert meta["name"] == "phase"
    assert meta["params"]["backend"] == "fft"


def test_kuramoto_block_output():
    series = [np.sin(np.linspace(0, np.pi, 200)), np.cos(np.linspace(0, np.pi, 200))]
    block = MultiAssetKuramotoBlock()
    result = block.transform(series)
    assert set(result.keys()) == {"phase", "kuramoto_order"}
    assert isinstance(result["phase"], list)
    assert isinstance(result["kuramoto_order"], float)
    assert result["kuramoto_order"] == multi_asset_kuramoto(series)


def test_kuramoto_order_feature_matches_function():
    phases = np.random.uniform(-np.pi, np.pi, size=(4, 128))
    feature = KuramotoOrderFeature(axis=0)
    assert np.allclose(feature.transform(phases), kuramoto_order(phases))
