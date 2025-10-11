"""High-performance numeric helpers with Rust accelerators and Python fallbacks."""

from __future__ import annotations

from typing import Sequence

import logging
import numpy as np

_logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional acceleration module
    from tradepulse_accel import (  # type: ignore
        convolve as _rust_convolve,
        quantiles as _rust_quantiles,
        sliding_windows as _rust_sliding_windows,
    )

    _RUST_ACCEL_AVAILABLE = True
except Exception:  # pragma: no cover - rust extension not built
    _rust_convolve = None  # type: ignore[assignment]
    _rust_quantiles = None  # type: ignore[assignment]
    _rust_sliding_windows = None  # type: ignore[assignment]
    _RUST_ACCEL_AVAILABLE = False


def _ensure_vector(data: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("input must be 1-dimensional")
    return arr


def _sliding_windows_python(arr: np.ndarray, window: int, step: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be greater than zero")
    if step <= 0:
        raise ValueError("step must be greater than zero")
    if arr.size < window:
        return np.empty((0, window), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(arr, window)
    if step != 1:
        view = view[::step]
    return np.array(view, copy=True)


def sliding_windows(
    data: Sequence[float] | np.ndarray,
    window: int,
    step: int = 1,
    *,
    use_rust: bool = True,
) -> np.ndarray:
    """Return a matrix of sliding windows over ``data``.

    Args:
        data: 1D input sequence.
        window: Size of each window (must be > 0).
        step: Step between windows (default: 1).
        use_rust: Attempt to dispatch to the Rust accelerator (default: True).

    Returns:
        ``(n_windows, window)`` matrix of float64 windows.
    """

    arr = _ensure_vector(data)
    if use_rust and _RUST_ACCEL_AVAILABLE and _rust_sliding_windows is not None:
        try:
            return _rust_sliding_windows(arr, int(window), int(step))
        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.warning(
                "Rust sliding_windows failed (%s); falling back to NumPy.",
                exc,
            )
    return _sliding_windows_python(arr, int(window), int(step))


def _quantiles_python(arr: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    probs = np.asarray(list(probabilities), dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D sequence")
    if np.any(~np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be within [0, 1]")
    if arr.size == 0:
        return np.full(probs.shape, np.nan, dtype=np.float64)
    return np.quantile(arr, probs, method="linear")


def quantiles(
    data: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    use_rust: bool = True,
) -> np.ndarray:
    """Compute quantiles for ``data`` at the given probabilities."""

    arr = _ensure_vector(data)
    if use_rust and _RUST_ACCEL_AVAILABLE and _rust_quantiles is not None:
        try:
            result = _rust_quantiles(arr, list(float(p) for p in probabilities))
            return np.asarray(result, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.warning(
                "Rust quantiles failed (%s); falling back to NumPy.",
                exc,
            )
    return _quantiles_python(arr, probabilities)


def _convolve_python(
    signal: np.ndarray,
    kernel: np.ndarray,
    *,
    mode: str = "full",
) -> np.ndarray:
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("convolution inputs must be 1-dimensional")
    return np.convolve(signal, kernel, mode=mode)


def convolve(
    signal: Sequence[float] | np.ndarray,
    kernel: Sequence[float] | np.ndarray,
    *,
    mode: str = "full",
    use_rust: bool = True,
) -> np.ndarray:
    """Convolve ``signal`` with ``kernel`` using the requested mode."""

    signal_arr = _ensure_vector(signal)
    kernel_arr = _ensure_vector(kernel)
    if use_rust and _RUST_ACCEL_AVAILABLE and _rust_convolve is not None:
        try:
            return _rust_convolve(signal_arr, kernel_arr, mode)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _logger.warning(
                "Rust convolve failed (%s); falling back to NumPy.",
                exc,
            )
    return _convolve_python(signal_arr, kernel_arr, mode=mode)


__all__ = ["sliding_windows", "quantiles", "convolve"]
