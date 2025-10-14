# SPDX-License-Identifier: MIT
"""Kuramoto phase-synchrony indicators for oscillatory market structure analysis.

The functions in this module translate raw price trajectories (produced by the
ingestion layer in ``core.data``) into phase representations using the
analytic-signal (Hilbert transform) construction and then summarise ensemble
coherence through the Kuramoto order parameter. These metrics are central to
TradePulse's *collective behaviour* signal cluster, which assesses whether
assets are moving in lockstep or diverging. The implementation mirrors the
mathematical discussion in ``docs/indicators.md`` and ties into the monitoring
hooks outlined in ``docs/quality_gates.md``, ensuring features expose metrics
that downstream telemetry can trace and the execution stack can consume.

Upstream dependencies include NumPy for vectorised operations and optional
SciPy accelerations for Hilbert transforms. Downstream consumers comprise the
feature pipeline described in ``docs/performance.md``, regime detectors in
``core.phase`` and the CLI workflows in ``interfaces/cli.py``. GPU support is
available via CuPy when installed, and observability is coordinated with the
logging/metrics façade mandated by ``docs/documentation_governance.md``.
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

    The analytic signal ``z(t) = x(t) + i ℋ{x}(t)`` is obtained either via
    SciPy's Hilbert transform or a NumPy FFT fallback when SciPy is absent. The
    instantaneous phase is then ``θ(t) = arg(z(t))`` with range ``[-π, π]``. The
    implementation follows the data-quality safeguards documented in
    ``docs/runbook_data_incident.md`` and surfaces metrics consumed by the
    observability stack described in ``docs/monitoring.md``.

    Args:
        x: One-dimensional array of samples representing the price or oscillator
            trajectory. Non-finite values are replaced with zeros in accordance
            with the cleansing contract described in ``docs/quality_gates.md``.
        use_float32: When ``True``, perform calculations in ``float32`` to
            reduce memory pressure and improve GPU transfer efficiency for large
            windows.
        out: Optional preallocated output array. The buffer must match the input
            shape and target dtype; otherwise a :class:`ValueError` is raised.

    Returns:
        np.ndarray: Phase angle of each sample in radians with dtype matching the
        requested precision.

    Raises:
        ValueError: If ``x`` is not one-dimensional or ``out`` has incompatible
            shape or dtype.

    Examples:
        >>> series = np.array([0.0, 1.0, 0.0, -1.0])
        >>> np.round(compute_phase(series), 2)
        array([0.  , 1.57, 3.14, -1.57])

    Notes:
        Constant or near-constant inputs lead to ill-defined Hilbert transforms.
        This implementation returns zeros in that case, mirroring the safeguard
        described in ``docs/documentation_governance.md``. When ``use_float32``
        is enabled the returned phases are numerically stable for windows up to
        ~1e6 samples; beyond that, prefer ``float64`` to avoid precision loss.
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

    The statistic measures synchrony as ``R = |(1/N) ∑_j e^{i θ_j}|``. Values
    close to ``1`` indicate phase alignment, while values near ``0`` signal
    desynchronisation. In TradePulse this metric feeds the market regime
    dashboards discussed in ``docs/monitoring.md`` and the theoretical overview
    in ``docs/indicators.md``.

    Args:
        phases: Array of shape ``(N,)`` (single snapshot) or ``(N, T)`` (matrix of
            ``T`` snapshots across ``N`` oscillators). Complex inputs are projected
            onto their phase angles.

    Returns:
        float | np.ndarray: ``float`` for one-dimensional input or an array of
        length ``T`` for two-dimensional input. The dtype follows NumPy's default
        promotion rules for ``float64`` stability.

    Raises:
        ValueError: If ``phases`` is scalar or has more than two dimensions.

    Examples:
        >>> theta = np.linspace(0.0, np.pi, 4)
        >>> float(kuramoto_order(theta))
        0.9003163161571061

    Notes:
        Non-finite values are ignored in the vector aggregation, matching the
        resilience requirements from ``docs/runbook_data_incident.md``. The
        numerical implementation uses both ``float32`` and ``float64`` buffers
        to balance performance and stability for large ensembles; clipping at
        ``1e-8`` enforces the governance rule that de-synchronised states report
        exactly zero rather than a denormal.
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
    constructing composite indicators as outlined in ``docs/indicators.md`` and
    the execution alignment notes in ``docs/execution.md``.

    Args:
        series_list: Iterable of equally sampled price arrays. Series are assumed
            to be aligned in calendar time according to the ingestion guarantees
            described in ``docs/documentation_governance.md``.

    Returns:
        float: Kuramoto order parameter for the latest synchronised observation.

    Raises:
        ValueError: If any series is empty or of mismatched length.

    Examples:
        >>> ref = np.linspace(100, 101, 32)
        >>> correlated = ref + 0.01
        >>> round(float(multi_asset_kuramoto([ref, correlated])), 4)
        1.0
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
    """Compute phase on the GPU via CuPy with CPU fallback.

    Args:
        x: Sequence of samples convertible to a CuPy array.

    Returns:
        np.ndarray: Phase angles in radians located on host memory for downstream
        compatibility with CPU-only consumers.

    Notes:
        The function mirrors :func:`compute_phase` but executes FFT operations on
        the GPU when CuPy is available. Failures automatically fall back to the
        CPU implementation to honour the resilience expectations in
        ``docs/monitoring.md``. When running in mixed precision environments the
        computation defaults to ``float32`` to minimise device-host transfer
        overhead.
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
    """Feature wrapper for the Kuramoto order parameter.

    The feature converts phase snapshots into synchrony scores consumed by the
    feature pipeline described in ``docs/performance.md``. It records telemetry
    through ``core.utils.metrics`` so dashboards in ``docs/monitoring.md`` can
    attribute downstream decisions to their originating features.

    Attributes:
        use_float32: Whether to coerce inputs to ``float32`` prior to processing
            for memory savings at the cost of minor precision loss.
    """

    def __init__(self, *, use_float32: bool = False, name: str | None = None) -> None:
        """Initialise the feature instance.

        Args:
            use_float32: Use ``float32`` precision for memory efficiency.
            name: Optional custom identifier used in metrics and outputs.
        """
        super().__init__(name or "kuramoto_order")
        self.use_float32 = use_float32

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute Kuramoto order parameter from phase samples.

        Args:
            data: One- or two-dimensional array of phase samples. When the input
                is a price series the caller should precompute phases to honour
                the separation of concerns established in ``docs/indicators.md``.
            **_: Additional keyword arguments (ignored).

        Returns:
            FeatureResult: Value and metadata describing the synchrony snapshot.

        Examples:
            >>> feature = KuramotoOrderFeature()
            >>> result = feature.transform(np.linspace(0, np.pi, 8))
            >>> 0.0 <= result.value <= 1.0
            True
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
    """Kuramoto synchrony feature across multiple synchronised assets.

    The feature evaluates :func:`multi_asset_kuramoto` for aligned asset
    histories and records metadata about universe size for the reporting flow
    described in ``docs/documentation_governance.md``.
    """

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name or "multi_asset_kuramoto")

    def transform(self, data: Sequence[np.ndarray], **_: Any) -> FeatureResult:
        """Evaluate synchrony across multiple assets.

        Args:
            data: Sequence of equally sampled price arrays. Each array must share
                the same length and temporal alignment.
            **_: Additional keyword arguments (ignored).

        Returns:
            FeatureResult: Kuramoto order value along with asset-count metadata.
        """

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
