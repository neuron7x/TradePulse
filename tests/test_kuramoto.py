# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    compute_phase_gpu,
    kuramoto_order,
    multi_asset_kuramoto,
)
from core.indicators import kuramoto as kuramoto_module


def test_kuramoto_range():
    x = np.sin(np.linspace(0, 10 * np.pi, 1000))
    ph = compute_phase(x)
    R = kuramoto_order(ph[-200:])
    assert 0.0 <= R <= 1.0


def test_compute_phase_validates_dimension():
    with pytest.raises(ValueError):
        compute_phase(np.zeros((4, 4)))


def test_compute_phase_gpu_falls_back_to_cpu():
    x = np.sin(np.linspace(0, 2 * np.pi, 128))
    cpu = compute_phase(x)
    gpu = compute_phase_gpu(x)
    assert gpu.shape == cpu.shape
    assert np.allclose(gpu, cpu, atol=1e-6)


def test_compute_phase_fft_branch_handles_odd_length():
    x = np.sin(np.linspace(0, 2 * np.pi, 129))
    phases = compute_phase(x)
    assert phases.shape == (129,)


def test_multi_asset_kuramoto_last_step_alignment():
    series = [
        np.sin(np.linspace(0, np.pi, 256)),
        np.sin(np.linspace(0, np.pi, 256) + np.pi / 4),
    ]
    value = multi_asset_kuramoto(series)
    assert 0.0 <= value <= 1.0


def test_compute_phase_uses_hilbert_when_available(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, int] = {"count": 0}

    def fake_hilbert(x):
        calls["count"] += 1
        return x + 1j * np.ones_like(x)

    monkeypatch.setattr(kuramoto_module, "hilbert", fake_hilbert)
    phases = kuramoto_module.compute_phase(np.ones(16))
    assert calls["count"] == 1
    assert phases.shape == (16,)


def test_kuramoto_order_handles_matrix_input():
    phases = np.vstack(
        [
            np.random.uniform(-np.pi, np.pi, size=64),
            np.random.uniform(-np.pi, np.pi, size=64),
        ]
    )
    values = kuramoto_order(phases)
    assert values.shape == (64,)
    assert np.all((0.0 <= values) & (values <= 1.0))


def test_compute_phase_gpu_with_fake_cupy(monkeypatch: pytest.MonkeyPatch):
    class _FakeFFT:
        @staticmethod
        def rfft(x, n):
            return np.fft.rfft(x, n)

        @staticmethod
        def irfft(x, n):
            return np.fft.irfft(x, n)

    class _FakeCP:
        float32 = np.float32
        fft = _FakeFFT()

        @staticmethod
        def asarray(x, dtype=None):
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def zeros_like(x):
            return np.zeros_like(x)

        @staticmethod
        def angle(x):
            return np.angle(x)

        @staticmethod
        def asnumpy(x):
            return np.asarray(x)

    monkeypatch.setattr(kuramoto_module, "cp", _FakeCP())
    phases_even = compute_phase_gpu(np.sin(np.linspace(0, 2 * np.pi, 64)))
    phases_odd = compute_phase_gpu(np.sin(np.linspace(0, 2 * np.pi, 65)))
    assert phases_even.shape == (64,)
    assert phases_odd.shape == (65,)


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
