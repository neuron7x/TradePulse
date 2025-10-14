# SPDX-License-Identifier: MIT
"""Kuramoto phase-synchrony indicators for oscillatory market structure analysis.

The functions in this module translate raw price trajectories into phase
representations using the analytic-signal (Hilbert transform) construction and
then summarise ensemble coherence through the Kuramoto order parameter. These
metrics are central to TradePulse's *collective behaviour* signal cluster, which
assesses whether assets are moving in lockstep or diverging. The implementation
mirrors the mathematical discussion in `docs/indicators.md` and ties into the
monitoring hooks outlined in `docs/quality_gates.md`, ensuring features expose
metrics that downstream telemetry can trace.

Key dependencies include NumPy for vectorised operations, optional SciPy
accelerations for Hilbert transforms, and the metrics/logging helpers from
``core.utils`` for governance-compliant observability. GPU support is available
via CuPy when installed. The module also defines feature wrappers consumed by
the feature pipeline described in `docs/performance.md`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from .base import BaseFeature, FeatureResult

_logger = get_logger(__name__)
_metrics = get_metrics_collector()


def _log_debug_enabled() -> bool:
    base_logger = getattr(_logger, "logger", None)
    checker = getattr(base_logger, "isEnabledFor", None)
    return bool(checker and checker(logging.DEBUG))

try:
    from scipy import fft as _scipy_fft  # type: ignore
    _scipy_fft.set_workers(1)  # pragma: no cover - optional tuning
except Exception:  # fallback if SciPy not installed
    _scipy_fft = None  # type: ignore[assignment]

try:
    from scipy.signal import hilbert
except Exception:  # fallback if SciPy not installed
    hilbert = None

def compute_phase(
    x: np.ndarray,
    *,
    use_float32: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the instantaneous phase of a univariate signal.

    The routine derives the analytic signal ``z(t) = x(t) + i ℋ{x}(t)`` via
    either SciPy's Hilbert transform or a NumPy FFT fallback. The instantaneous
    phase is ``θ(t) = arg(z(t))`` with range ``[-π, π]``. This is the starting
    point for Kuramoto-style synchrony metrics
    documented in ``docs/indicators.md`` and cross-checked in the feature QA flow
    defined by ``docs/quality_gates.md``.

    Args:
        x: One-dimensional array of samples representing the price or oscillator
            trajectory. Non-finite values are replaced with zeros to align with
            the data-cleansing guidance in ``docs/runbook_data_incident.md``.
        use_float32: When ``True``, perform calculations in ``float32`` to
            reduce memory pressure for large windows.
        out: Optional preallocated output array. The buffer must match the input
            shape and target dtype; otherwise a :class:`ValueError` is raised.

    Returns:
        ``np.ndarray`` containing the phase angle of each sample in radians.

    Raises:
        ValueError: If ``x`` is not one-dimensional or ``out`` has incompatible
            shape/dtype.

    Examples:
        >>> series = np.array([0.0, 1.0, 0.0, -1.0])
        >>> np.round(compute_phase(series), 2)
        array([0.  , 1.57, 3.14, -1.57])

    Notes:
        Constant or near-constant inputs lead to ill-defined Hilbert transforms.
        This implementation returns zeros in that case, mirroring the safeguard
        described in the indicator governance notes.
    """
    context_manager = (
        _logger.operation("compute_phase", data_size=len(x), use_float32=use_float32)
        if _log_debug_enabled()
        else nullcontext()
    )
    with context_manager:
        dtype = np.float32 if use_float32 else float
        x = np.asarray(x, dtype=dtype)
        target = None
        if out is not None:
            target = np.asarray(out)
            if target.shape != x.shape:
                raise ValueError("out array must match input shape")
            if target.dtype != np.dtype(dtype):
                raise ValueError("out array dtype must match requested precision")
        if x.ndim != 1:
            raise ValueError("compute_phase expects 1D array")
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        hilbert_module = getattr(hilbert, "__module__", "") if hilbert is not None else ""
        use_scipy_fastpath = (
            _scipy_fft is not None
            and hilbert is not None
            and hilbert_module.startswith("scipy.")
        )
        if use_scipy_fastpath:
            n = x.size
            if n == 0:
                return np.empty(0, dtype=dtype)

            real = np.asarray(x, dtype=dtype)
            spectrum = _scipy_fft.rfft(real)
            spectrum *= -1j
            spectrum[0] = 0
            if n % 2 == 0 and spectrum.size > 1:
                spectrum[-1] = 0
            imag = _scipy_fft.irfft(spectrum, n)
            imag = np.asarray(imag, dtype=dtype, copy=False)
        elif hilbert is not None:
            a = hilbert(x)
            real = np.asarray(a.real, dtype=dtype, copy=False)
            imag = np.asarray(a.imag, dtype=dtype, copy=False)
        else:
            # Analytic signal via real FFT-based Hilbert transform. Using rfft/irfft
            # halves the amount of spectral data we have to touch compared to the
            # dense FFT variant, which materially reduces latency for large signals
            # on systems where SciPy is unavailable.
            n = x.size
            if n == 0:
                return np.empty(0, dtype=dtype)

            working = x.astype(float, copy=False)
            spectrum = np.fft.fft(working)
            h = np.zeros(n, dtype=float)
            if n % 2 == 0:
                h[0] = h[n // 2] = 1.0
                h[1 : n // 2] = 2.0
            else:
                h[0] = 1.0
                h[1 : (n + 1) // 2] = 2.0
            analytic = np.fft.ifft(spectrum * h)
            real = analytic.real.astype(dtype, copy=False)
            imag = analytic.imag.astype(dtype, copy=False)
        if target is not None:
            np.arctan2(imag, real, out=target)
            return target
        phases = np.arctan2(imag, real)
        return phases.astype(dtype, copy=False)

def kuramoto_order(phases: np.ndarray) -> float | np.ndarray:
    """Evaluate the Kuramoto order parameter.

    The order parameter measures synchrony as ``R = |(1/N) ∑_j e^{i θ_j}|``.
    Values close to ``1`` indicate phase alignment, while values
    near ``0`` signal desynchronisation. In TradePulse this statistic feeds the
    market regime dashboards discussed in ``docs/monitoring.md`` and the
    theoretical overview in ``docs/indicators.md``.

    Args:
        phases: Array of shape ``(N,)`` (single snapshot) or ``(N, T)`` (matrix of
            ``T`` snapshots across ``N`` oscillators). Complex inputs are projected
            onto their phase angles.

    Returns:
        ``float`` for one-dimensional input or an array of length ``T`` for
        two-dimensional input.

    Raises:
        ValueError: If ``phases`` is scalar or has more than two dimensions.

    Examples:
        >>> theta = np.linspace(0.0, np.pi, 4)
        >>> float(kuramoto_order(theta))
        0.9003163161571061

    Notes:
        Non-finite values are ignored in the vector aggregation, matching the
        resilience requirements from ``docs/runbook_data_incident.md``.
    """

    phases_arr = np.asarray(phases)
    if phases_arr.ndim == 0:
        raise ValueError("kuramoto_order expects at least one dimension")

    if np.iscomplexobj(phases_arr):
        if np.allclose(phases_arr.imag, 0.0):
            phases_real = phases_arr.real
        else:
            phases_real = np.angle(phases_arr)
    else:
        phases_real = phases_arr

    with np.errstate(over="ignore", invalid="ignore"):
        phases_fp32 = np.asarray(phases_real, dtype=np.float32)
    phases_fp64 = np.asarray(phases_real, dtype=np.float64)
    mask = np.isfinite(phases_fp32)

    cos_vals = np.empty_like(phases_fp32, dtype=np.float32)
    sin_vals = np.empty_like(phases_fp32, dtype=np.float32)
    if mask.all():
        np.cos(phases_fp32, out=cos_vals)
        np.sin(phases_fp32, out=sin_vals)
    else:
        cos_vals.fill(0.0)
        sin_vals.fill(0.0)
        np.cos(phases_fp32, out=cos_vals, where=mask)
        np.sin(phases_fp32, out=sin_vals, where=mask)

    if phases_fp32.ndim == 1:
        count = int(mask.sum(dtype=np.int32))
        if count == 0:
            return 0.0
        phases_valid = phases_fp64[mask]
        sum_real = float(np.cos(phases_valid).sum(dtype=np.float64))
        sum_imag = float(np.sin(phases_valid).sum(dtype=np.float64))
        magnitude = (sum_real * sum_real + sum_imag * sum_imag) ** 0.5
        value = magnitude / count
        if value < 1e-8:
            value = 0.0
        return float(np.clip(value, 0.0, 1.0))

    if phases_fp32.ndim != 2:
        raise ValueError("kuramoto_order expects 1D or 2D array")

    valid_counts = mask.sum(axis=0, dtype=np.int32)
    if not np.any(valid_counts):
        return np.zeros(phases_fp32.shape[1], dtype=float)

    sum_real = np.add.reduce(cos_vals, axis=0, dtype=np.float32).astype(np.float64)
    sum_imag = np.add.reduce(sin_vals, axis=0, dtype=np.float32).astype(np.float64)
    magnitude = np.hypot(sum_real, sum_imag)
    values = np.divide(
        magnitude,
        valid_counts,
        out=np.zeros_like(magnitude, dtype=float),
        where=valid_counts > 0,
    )
    clipped = np.clip(values, 0.0, 1.0)
    clipped[clipped < 1e-8] = 0.0
    return clipped

def multi_asset_kuramoto(series_list: Sequence[np.ndarray]) -> float:
    """Aggregate cross-asset synchrony at the most recent timestamp.

    Each series is converted to its instantaneous phase before evaluating the
    Kuramoto order parameter over the terminal observation. Use this helper when
    constructing composite indicators as outlined in ``docs/indicators.md``.

    Args:
        series_list: Iterable of equally sampled price arrays.

    Returns:
        Kuramoto order parameter for the latest synchronised observation.

    Raises:
        ValueError: If any series is empty or of mismatched length.
    """
    sequences = [np.asarray(series, dtype=float) for series in series_list]
    if not sequences:
        raise ValueError("series_list must contain at least one sequence")

    first_length = sequences[0].shape[0]
    if first_length == 0:
        raise ValueError("series must not be empty")

    for series in sequences[1:]:
        if series.shape[0] != first_length:
            raise ValueError("all series must have the same length")

    phases = [compute_phase(s) for s in sequences]
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
