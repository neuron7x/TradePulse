# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.indicators.kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    compute_phase_gpu,
    kuramoto_order,
    multi_asset_kuramoto,
)


def test_compute_phase_matches_expected_linear_phase(sin_wave: np.ndarray) -> None:
    phase = compute_phase(sin_wave)
    t = np.linspace(0, 4 * np.pi, sin_wave.size, endpoint=False)
    expected = np.unwrap(t - np.pi / 2)
    np.testing.assert_allclose(
        np.unwrap(phase),
        expected,
        atol=5e-2,
        err_msg="Instantaneous phase should follow analytical sine phase profile",
    )


def test_compute_phase_requires_one_dimensional_input() -> None:
    with pytest.raises(ValueError):
        compute_phase(np.ones((4, 4)))


def test_kuramoto_order_is_one_for_aligned_phases() -> None:
    phases = np.zeros(128)
    result = kuramoto_order(phases)
    assert pytest.approx(1.0, rel=1e-12) == result


def test_kuramoto_order_handles_matrix_input() -> None:
    phases = np.vstack([np.zeros(16), np.pi * np.ones(16)])
    result = kuramoto_order(phases)
    assert result.shape == (16,)
    assert np.all((0.0 <= result) & (result <= 1.0))


def test_multi_asset_kuramoto_uses_last_phase_alignment() -> None:
    base = np.linspace(0, 6 * np.pi, 256, endpoint=False)
    series_a = np.sin(base)
    series_b = np.sin(base + 0.2)
    result = multi_asset_kuramoto([series_a, series_b])
    phase_a = compute_phase(series_a)[-1]
    phase_b = compute_phase(series_b)[-1]
    reference = kuramoto_order(np.array([phase_a, phase_b]))
    assert pytest.approx(reference, rel=1e-5) == result


def test_compute_phase_gpu_fallback_matches_cpu() -> None:
    data = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
    cpu = compute_phase(data)
    gpu = compute_phase_gpu(data)
    np.testing.assert_allclose(cpu, gpu, atol=1e-6)


def test_compute_phase_handles_odd_length_series() -> None:
    data = np.sin(np.linspace(0, 2 * np.pi, 129, endpoint=False))
    phase = compute_phase(data)
    assert phase.shape == data.shape
    assert np.isfinite(phase).all()


def test_compute_phase_uses_custom_hilbert(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.indicators.kuramoto as module

    data = np.linspace(-np.pi, np.pi, 16, endpoint=False)

    def fake_hilbert(values: np.ndarray) -> np.ndarray:
        return np.exp(1j * values)

    monkeypatch.setattr(module, "hilbert", fake_hilbert)
    phase = module.compute_phase(data)
    np.testing.assert_allclose(np.unwrap(phase), np.unwrap(data), atol=1e-9)


def test_compute_phase_gpu_uses_gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.indicators.kuramoto as module

    def _asarray(values, dtype=None):
        if dtype is None:
            return np.asarray(values)
        return np.asarray(values, dtype=dtype)

    fake_cp = SimpleNamespace(
        asarray=_asarray,
        float32=np.float32,
        zeros=lambda n, dtype=None: np.zeros(n, dtype=dtype if dtype is not None else np.float32),
        fft=SimpleNamespace(
            fft=lambda x: np.fft.fft(_asarray(x)),
            ifft=lambda x: np.fft.ifft(_asarray(x)),
        ),
        angle=np.angle,
        asnumpy=lambda x: np.asarray(x),
    )

    monkeypatch.setattr(module, "cp", fake_cp)

    even = np.sin(np.linspace(0, 2 * np.pi, 128, endpoint=False))
    even_gpu = module.compute_phase_gpu(even)
    even_cpu = module.compute_phase(even)
    np.testing.assert_allclose(np.unwrap(even_gpu), np.unwrap(even_cpu), atol=1e-6)

    odd = np.sin(np.linspace(0, 2 * np.pi, 129, endpoint=False))
    odd_gpu = module.compute_phase_gpu(odd)
    odd_cpu = module.compute_phase(odd)
    np.testing.assert_allclose(np.unwrap(odd_gpu), np.unwrap(odd_cpu), atol=1e-6)


def test_kuramoto_order_feature_returns_expected_metadata() -> None:
    feature = KuramotoOrderFeature()
    result = feature.transform(np.zeros(32))
    assert result.name == "kuramoto_order"
    assert result.metadata == {}
    assert result.value == pytest.approx(1.0, rel=1e-12)


def test_multi_asset_kuramoto_feature_reports_asset_count(sin_wave: np.ndarray) -> None:
    feature = MultiAssetKuramotoFeature(name="multi")
    data = [sin_wave, np.roll(sin_wave, 3)]
    outcome = feature.transform(data)
    assert outcome.name == "multi"
    assert outcome.metadata == {"assets": 2}
    expected = multi_asset_kuramoto(data)
    assert outcome.value == pytest.approx(expected, rel=1e-12)
