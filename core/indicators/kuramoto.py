# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import ArrayLike, BaseBlock, BaseFeature

try:
    from scipy.signal import hilbert
except Exception:  # fallback if SciPy not installed
    hilbert = None

__all__ = [
    "PhaseFeature",
    "KuramotoOrderFeature",
    "MultiAssetKuramotoBlock",
    "compute_phase",
    "kuramoto_order",
    "multi_asset_kuramoto",
    "compute_phase_gpu",
]


class PhaseFeature(BaseFeature):
    """Estimate the instantaneous phase of a 1-D signal."""

    def __init__(self, backend: str = "auto") -> None:
        backend = backend.lower()
        if backend not in {"auto", "hilbert", "fft"}:
            raise ValueError("backend must be 'auto', 'hilbert' or 'fft'")
        super().__init__(
            name="phase",
            params={"backend": backend},
            description="Analytic signal phase computed via Hilbert transform or FFT fallback.",
        )
        self._backend = backend

    def transform(self, series: ArrayLike) -> np.ndarray:
        x = self.coerce_vector(series, allow_empty=False)
        backend = self._backend
        if backend == "auto":
            backend = "hilbert" if hilbert is not None else "fft"

        if backend == "hilbert":
            if hilbert is None:
                raise RuntimeError("SciPy is required for the Hilbert backend")
            analytic = hilbert(x)
        else:
            n = x.size
            X = np.fft.rfft(x, n)
            H = np.zeros_like(X)
            if n % 2 == 0:
                H[1:-1] = 2
            else:
                H[1:] = 2
            imag = np.fft.irfft(X * H, n)
            analytic = x + 1j * imag
        return np.angle(analytic)


class KuramotoOrderFeature(BaseFeature):
    """Kuramoto order parameter :math:`R` from a collection of phases."""

    def __init__(self, axis: int | None = None) -> None:
        super().__init__(
            name="kuramoto_order",
            params={"axis": axis},
            description="Magnitude of the mean phase vector across oscillators.",
        )
        self._axis = axis

    def transform(self, phases: ArrayLike | np.ndarray) -> float | np.ndarray:
        z = np.exp(1j * np.asarray(phases, dtype=float))
        if z.ndim == 0:
            return float(np.abs(z))
        if z.ndim == 1:
            return float(np.abs(np.mean(z)))
        if z.ndim == 2:
            axis = 0 if self._axis is None else self._axis
            return np.abs(np.mean(z, axis=axis)).astype(float)
        raise ValueError("KuramotoOrderFeature expects 1-D or 2-D input")


class MultiAssetKuramotoBlock(BaseBlock):
    """Compute instantaneous phases and Kuramoto order across assets."""

    def __init__(
        self,
        phase_feature: PhaseFeature | None = None,
        order_feature: KuramotoOrderFeature | None = None,
    ) -> None:
        self._phase_feature = phase_feature or PhaseFeature()
        self._order_feature = order_feature or KuramotoOrderFeature()
        super().__init__(
            name="multi_asset_kuramoto",
            features=(self._phase_feature, self._order_feature),
            description=(
                "Transforms synchronised price series to instantaneous phases "
                "and aggregates them via the Kuramoto order parameter."
            ),
        )

    def transform(self, series_list: Sequence[ArrayLike]) -> dict[str, np.ndarray | float]:
        if not series_list:
            raise ValueError("MultiAssetKuramotoBlock requires at least one series")
        phases = [self._phase_feature.transform(series) for series in series_list]
        last_phases = np.array([phase[-1] for phase in phases], dtype=float)
        order_value = self._order_feature.transform(last_phases)
        return {
            self._phase_feature.name: phases,
            self._order_feature.name: order_value,
        }


def compute_phase(x: ArrayLike) -> np.ndarray:
    """Functional wrapper for :class:`PhaseFeature`."""

    return PhaseFeature().transform(x)


def kuramoto_order(phases: ArrayLike | np.ndarray) -> float | np.ndarray:
    """Functional wrapper for :class:`KuramotoOrderFeature`."""

    return KuramotoOrderFeature().transform(phases)


def multi_asset_kuramoto(series_list: Sequence[ArrayLike]) -> float:
    """Functional wrapper for :class:`MultiAssetKuramotoBlock`."""

    block = MultiAssetKuramotoBlock()
    result = block.transform(series_list)
    return float(result["kuramoto_order"])


# Optional GPU acceleration via CuPy (if available)
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None


def compute_phase_gpu(x: ArrayLike) -> np.ndarray:
    """GPU phase via CuPy Hilbert transform if CuPy is available; else falls back."""

    if cp is None:
        return PhaseFeature().transform(x)
    x_gpu = cp.asarray(x, dtype=cp.float32)
    n = x_gpu.size
    X = cp.fft.rfft(x_gpu, n)
    H = cp.zeros_like(X)
    if n % 2 == 0:
        H[1:-1] = 2
    else:
        H[1:] = 2
    a = cp.fft.irfft(X * H, n)
    analytic = x_gpu + 1j * a
    ph = cp.angle(analytic)
    return cp.asnumpy(ph)
