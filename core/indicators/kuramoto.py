# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseFeature, FeatureResult

try:
    from scipy.signal import hilbert
except Exception:  # fallback if SciPy not installed
    hilbert = None

def compute_phase(x: np.ndarray) -> np.ndarray:
    """Compute instantaneous phase via Hilbert transform.
    If SciPy unavailable, fall back to angle of analytic signal via FFT trick.
    Args:
        x: 1D array of samples.
    Returns:
        phases in radians in [-pi, pi].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("compute_phase expects 1D array")
    if hilbert is not None:
        a = hilbert(x)
    else:
        # crude analytic signal via FFT (imag as Hilbert by signum in freq)
        n = x.size
        X = np.fft.rfft(x, n)
        H = np.zeros_like(X)
        if n % 2 == 0:
            H[1:-1] = 2
        else:
            H[1:] = 2
        a = np.fft.irfft(X * H, n)
        a = x + 1j * a
    return np.angle(a)

def kuramoto_order(phases: np.ndarray) -> float:
    """Kuramoto order parameter R = |mean(exp(iθ))| for a set of phases.
    Args:
        phases: array of shape (N,) or (N,T). If 1D, compute for one time slice;
                if 2D, compute per-column over N oscillators.
    Returns:
        float if 1D else np.ndarray of shape (T,).
    """
    z = np.exp(1j * phases)
    if z.ndim == 1:
        return float(np.abs(np.mean(z)))
    return np.abs(np.mean(z, axis=0)).astype(float)

def multi_asset_kuramoto(series_list: Sequence[np.ndarray]) -> float:
    """Compute Kuramoto R across multiple synchronized assets (same length).
    Each series is transformed to phase, then R over assets for the last time step.
    """
    phases = [compute_phase(s) for s in series_list]
    last_phases = np.array([p[-1] for p in phases])
    return kuramoto_order(last_phases)


# Optional GPU acceleration via CuPy (if available)
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

def compute_phase_gpu(x):
    """GPU phase via CuPy Hilbert transform if CuPy is available; else falls back."""
    if cp is None:
        from .kuramoto import compute_phase as _cpu
        return _cpu(np.asarray(x))
    x_gpu = cp.asarray(x, dtype=cp.float32)
    # simple Hilbert via FFT sign filter on GPU
    n = x_gpu.size
    X = cp.fft.rfft(x_gpu, n)
    H = cp.zeros_like(X)
    if n % 2 == 0:
        H[1:-1] = 2
    else:
        H[1:] = 2
    a = cp.fft.irfft(X * H, n)
    a = x_gpu + 1j * a
    ph = cp.angle(a)
    return cp.asnumpy(ph)


class KuramotoOrderFeature(BaseFeature):
    """Feature wrapper for the Kuramoto order parameter."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name or "kuramoto_order")

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        value = kuramoto_order(data)
        return FeatureResult(name=self.name, value=value, metadata={})


class MultiAssetKuramotoFeature(BaseFeature):
    """Feature computing the Kuramoto order across multiple synchronized assets."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name or "multi_asset_kuramoto")

    def transform(self, data: Sequence[np.ndarray], **_: Any) -> FeatureResult:
        value = multi_asset_kuramoto(data)
        metadata = {"assets": len(data)}
        return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = [
    "compute_phase",
    "compute_phase_gpu",
    "kuramoto_order",
    "multi_asset_kuramoto",
    "KuramotoOrderFeature",
    "MultiAssetKuramotoFeature",
]
