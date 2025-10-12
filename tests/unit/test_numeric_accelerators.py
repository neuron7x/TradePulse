from __future__ import annotations

import numpy as np
import pytest

from core.accelerators import convolve, quantiles, sliding_windows
from core.accelerators.numeric import (
    convolve_numpy_backend,
    convolve_python_backend,
    numpy_available,
    quantiles_numpy_backend,
    quantiles_python_backend,
    rust_available,
    sliding_windows_numpy_backend,
    sliding_windows_python_backend,
)


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


def test_python_backends_match_public_api() -> None:
    data = [0.0, 1.0, 2.0, 3.0]
    windows_py = sliding_windows_python_backend(data, 2, 1)
    quantiles_py = quantiles_python_backend(data, (0.25, 0.75))
    conv_py = convolve_python_backend(data, [1.0, -1.0], mode="full")

    expected_windows = sliding_windows(data, window=2, step=1, use_rust=False)
    expected_quantiles = quantiles(data, (0.25, 0.75), use_rust=False)
    expected_convolve = convolve(data, [1.0, -1.0], mode="full", use_rust=False)

    np.testing.assert_allclose(np.asarray(windows_py, dtype=float), expected_windows)
    np.testing.assert_allclose(np.asarray(quantiles_py, dtype=float), expected_quantiles)
    np.testing.assert_allclose(np.asarray(conv_py, dtype=float), expected_convolve)


@pytest.mark.skipif(not numpy_available(), reason="NumPy backend not available")
def test_numpy_backends_match_public_api() -> None:
    data = np.linspace(-1.0, 1.0, 8)
    probs = (0.1, 0.5, 0.9)
    kernel = np.array([1.0, 0.5, -0.5], dtype=float)

    windows_np = sliding_windows_numpy_backend(data, 3, 2)
    quantiles_np = quantiles_numpy_backend(data, probs)
    convolve_np = convolve_numpy_backend(data, kernel, mode="same")

    expected_windows = sliding_windows(data, window=3, step=2, use_rust=False)
    expected_quantiles = quantiles(data, probs, use_rust=False)
    expected_convolve = convolve(data, kernel, mode="same", use_rust=False)

    np.testing.assert_allclose(windows_np, expected_windows)
    np.testing.assert_allclose(quantiles_np, expected_quantiles)
    np.testing.assert_allclose(convolve_np, expected_convolve)


def test_backend_availability_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.accelerators.numeric as numeric

    assert isinstance(numeric.numpy_available(), bool)
    assert isinstance(numeric.rust_available(), bool)

    monkeypatch.setattr(numeric, "_NUMPY_AVAILABLE", False)
    monkeypatch.setattr(numeric, "np", None)
    monkeypatch.setattr(numeric, "_RUST_ACCEL_AVAILABLE", False)
    monkeypatch.setattr(numeric, "_rust_convolve", None)
    monkeypatch.setattr(numeric, "_rust_quantiles", None)
    monkeypatch.setattr(numeric, "_rust_sliding_windows", None)

    assert numeric.numpy_available() is False
    assert numeric.rust_available() is False


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
