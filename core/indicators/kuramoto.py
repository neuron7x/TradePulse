# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseFeature, FeatureResult
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

_logger = get_logger(__name__)
_metrics = get_metrics_collector()

try:
    from scipy.signal import hilbert
except Exception:  # fallback if SciPy not installed
    hilbert = None

def compute_phase(x: np.ndarray, *, use_float32: bool = False) -> np.ndarray:
    """Compute instantaneous phase via Hilbert transform.
    If SciPy unavailable, fall back to angle of analytic signal via FFT trick.
    Args:
        x: 1D array of samples.
        use_float32: Use float32 precision to reduce memory usage (default: False)
    Returns:
        phases in radians in [-pi, pi].
    """
    with _logger.operation("compute_phase", data_size=len(x), use_float32=use_float32):
        dtype = np.float32 if use_float32 else float
        x = np.asarray(x, dtype=dtype)
        if x.ndim != 1:
            raise ValueError("compute_phase expects 1D array")
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if hilbert is not None:
            a = hilbert(x)
        else:
            # Analytic signal via frequency-domain Hilbert transform.
            n = x.size
            X = np.fft.fft(x)
            H = np.zeros(n, dtype=dtype)
            if n % 2 == 0:
                H[0] = H[n // 2] = 1
                H[1 : n // 2] = 2
            else:
                H[0] = 1
                H[1 : (n + 1) // 2] = 2
            a = np.fft.ifft(X * H)
        return np.angle(a)

def kuramoto_order(phases: np.ndarray) -> float | np.ndarray:
    """Kuramoto order parameter R = |mean(exp(iθ))| for a set of phases.

    Args:
        phases: array of shape ``(N,)`` or ``(N, T)``. If 1D, compute for one time
            slice; if 2D, compute per-column over ``N`` oscillators.

    Returns:
        ``float`` if 1D else ``np.ndarray`` of shape ``(T,)``.
    """

    phases_arr = np.asarray(phases)
    if phases_arr.ndim == 0:
        raise ValueError("kuramoto_order expects at least one dimension")

    if np.iscomplexobj(phases_arr):
        # Support callers that accidentally pass analytic signals or complex
        # phases. Preserve real-only values to avoid the branch-cut at zero.
        if np.allclose(phases_arr.imag, 0.0):
            phases_real = phases_arr.real.astype(float, copy=False)
        else:
            phases_real = np.angle(phases_arr)
    else:
        phases_real = phases_arr.astype(float, copy=False)

    mask = np.isfinite(phases_real)
    safe_phases = np.where(mask, phases_real, 0.0)
    z = np.exp(1j * safe_phases)
    z = np.where(mask, z, np.nan + 0j)

    if z.ndim == 1:
        if not mask.any():
            return 0.0
        value = float(np.abs(np.nanmean(z)))
        # Numerical noise can push the magnitude above one by ~1e-16; clamp to unit
        # circle bounds so downstream users do not have to special-case tolerance.
        return float(np.clip(value, 0.0, 1.0))

    if z.ndim != 2:
        raise ValueError("kuramoto_order expects 1D or 2D array")

    valid_counts = mask.sum(axis=0)
    if not valid_counts.any():
        return np.zeros(z.shape[1], dtype=float)

    values = np.abs(np.nanmean(z, axis=0))
    values = np.where(valid_counts == 0, 0.0, values)
    return np.clip(values.astype(float, copy=False), 0.0, 1.0)

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
    """GPU phase via CuPy Hilbert transform if CuPy is available; else falls back.
    
    This function properly handles CuPy imports and memory transfers, ensuring
    correct GPU utilization when CuPy is available.
    """
    with _logger.operation("compute_phase_gpu", data_size=len(x), has_cupy=cp is not None):
        if cp is None:
            _logger.info("CuPy not available, falling back to CPU compute_phase")
            return compute_phase(np.asarray(x))
        
        try:
            # Use float32 for GPU efficiency
            x_gpu = cp.asarray(x, dtype=cp.float32)
            # Analytic signal via the same FFT approach as the CPU fallback.
            n = x_gpu.size
            X = cp.fft.fft(x_gpu)
            H = cp.zeros(n, dtype=cp.float32)
            if n % 2 == 0:
                H[0] = H[n // 2] = 1
                H[1 : n // 2] = 2
            else:
                H[0] = 1
                H[1 : (n + 1) // 2] = 2
            a = cp.fft.ifft(X * H)
            ph = cp.angle(a)
            return cp.asnumpy(ph)
        except Exception as e:
            _logger.warning(f"GPU computation failed, falling back to CPU: {e}")
            return compute_phase(np.asarray(x))


class KuramotoOrderFeature(BaseFeature):
    """Feature wrapper for the Kuramoto order parameter."""

    def __init__(self, *, use_float32: bool = False, name: str | None = None) -> None:
        """Initialize Kuramoto order parameter feature.
        
        Args:
            use_float32: Use float32 precision for memory efficiency (default: False)
            name: Optional custom name (default: "kuramoto_order")
        """
        super().__init__(name or "kuramoto_order")
        self.use_float32 = use_float32

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute Kuramoto order parameter.
        
        Args:
            data: 1D or 2D array of phases or time series
            **_: Additional keyword arguments (ignored)
            
        Returns:
            FeatureResult containing Kuramoto order parameter and metadata
        """
        with _metrics.measure_feature_transform(self.name, "kuramoto"):
            # Convert to appropriate dtype if needed
            if self.use_float32:
                data = np.asarray(data, dtype=np.float32)
            value = kuramoto_order(data)
            _metrics.record_feature_value(self.name, value)
            metadata: dict[str, Any] = {}
            if self.use_float32:
                metadata["use_float32"] = True

            return FeatureResult(
                name=self.name,
                value=value,
                metadata=metadata,
            )


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
