# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.kuramoto import (
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
