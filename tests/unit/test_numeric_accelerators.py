from __future__ import annotations

import numpy as np
import pytest

from core.accelerators import convolve, quantiles, sliding_windows


def test_sliding_windows_matches_numpy() -> None:
    data = np.linspace(0.0, 1.0, 16)
    expected = sliding_windows(data, window=4, step=2, use_rust=False)
    result = sliding_windows(data, window=4, step=2, use_rust=True)
    np.testing.assert_allclose(result, expected)


def test_quantiles_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(size=128)
    probabilities = (0.1, 0.5, 0.9)
    expected = quantiles(data, probabilities, use_rust=False)
    result = quantiles(data, probabilities, use_rust=True)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_matches_numpy(mode: str) -> None:
    signal = np.array([0.5, 1.0, -0.5, 2.0, 0.0], dtype=float)
    kernel = np.array([1.0, -1.0, 0.5], dtype=float)
    expected = convolve(signal, kernel, mode=mode, use_rust=False)
    result = convolve(signal, kernel, mode=mode, use_rust=True)
    np.testing.assert_allclose(result, expected)


def test_convolve_rejects_multidimensional_inputs() -> None:
    signal = np.ones((2, 2), dtype=float)
    kernel = np.array([1.0, 1.0], dtype=float)
    with pytest.raises(ValueError):
        convolve(signal, kernel)


@pytest.mark.parametrize("func_name", ["sliding_windows", "quantiles", "convolve"])
def test_rust_extension_matches_numpy_when_available(func_name: str) -> None:
    accel = pytest.importorskip("tradepulse_accel")
    rng = np.random.default_rng(7)

    if func_name == "sliding_windows":
        data = rng.normal(size=32)
        expected = sliding_windows(data, window=5, step=3, use_rust=False)
        result = accel.sliding_windows(np.asarray(data, dtype=np.float64), 5, 3)
        np.testing.assert_allclose(result, expected)
    elif func_name == "quantiles":
        data = rng.normal(size=64)
        probs = [0.2, 0.4, 0.8]
        expected = quantiles(data, probs, use_rust=False)
        result = np.asarray(accel.quantiles(np.asarray(data, dtype=np.float64), probs))
        np.testing.assert_allclose(result, expected)
    else:
        signal = rng.normal(size=16)
        kernel = rng.normal(size=5)
        expected = convolve(signal, kernel, mode="same", use_rust=False)
        result = accel.convolve(
            np.asarray(signal, dtype=np.float64),
            np.asarray(kernel, dtype=np.float64),
            "same",
        )
        np.testing.assert_allclose(result, expected)
