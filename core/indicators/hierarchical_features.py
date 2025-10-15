"""Hierarchical feature computation for multi-timeframe analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.data.resampling import align_timeframes
from .hurst import hurst_exponent
from .kuramoto import compute_phase


@dataclass(frozen=True)
class TimeFrameSpec:
    """Descriptor for each timeframe used in the hierarchy."""

    name: str
    frequency: str


@dataclass
class FeatureBufferCache:
    """Cache float32 buffers to avoid repeated allocations."""

    store: MutableMapping[str, np.ndarray] = field(default_factory=dict)

    def array(self, key: str, values: Sequence[float]) -> np.ndarray:
        src = np.asarray(values, dtype=np.float32)
        existing = self.store.get(key)
        if existing is None or existing.shape != src.shape:
            existing = np.empty_like(src)
            self.store[key] = existing
        np.copyto(existing, src, casting="unsafe")
        return existing

    def buffer(
        self,
        key: str,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        arr = self.store.get(key)
        if arr is None or arr.shape != shape or arr.dtype != np.dtype(dtype):
            arr = np.empty(shape, dtype=dtype)
            self.store[key] = arr
        return arr


def _shannon_entropy(series: np.ndarray, bins: int = 30) -> float:
    values = series[np.isfinite(series)]
    if values.size == 0:
        return 0.0

    values = np.asarray(values, dtype=np.float32)

    max_abs = float(np.max(np.abs(values)))
    if max_abs and np.isfinite(max_abs):
        values = values / max_abs

    counts, _ = np.histogram(values, bins=bins, density=False)
    total = counts.sum()
    if total <= 0:
        return 0.0

    probs = counts[counts > 0].astype(np.float32, copy=False)
    if probs.size == 0:
        return 0.0

    total = np.float32(total)
    inv_total = np.float32(1.0) / total
    np.multiply(probs, inv_total, out=probs)

    log_probs = np.log(probs, out=np.empty_like(probs))
    np.multiply(probs, log_probs, out=log_probs)
    entropy = np.add.reduce(log_probs, dtype=np.float32)
    return float(-entropy)


@dataclass
class HierarchicalFeatureResult:
    """Structured container for hierarchical feature outputs."""

    features: Dict[str, Dict[str, float]]
    multi_tf_phase_coherence: float
    benchmarks: Dict[str, float]


def _flatten(features: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {f"{tf}.{name}": value for tf, values in features.items() for name, value in values.items()}


def compute_hierarchical_features(
    ohlcv_by_tf: Mapping[str, pd.DataFrame],
    *,
    book_by_tf: Optional[Mapping[str, pd.DataFrame]] = None,
    benchmarks: Optional[Mapping[str, float]] = None,
    cache: Optional[FeatureBufferCache] = None,
) -> HierarchicalFeatureResult:
    """Compute Kuramoto, entropy, Hurst, book imbalance and microprice metrics."""

    if not ohlcv_by_tf:
        raise ValueError("ohlcv_by_tf must not be empty")
    cache = cache or FeatureBufferCache()
    reference = next(iter(ohlcv_by_tf))
    aligned = align_timeframes(ohlcv_by_tf, reference=reference)
    features: Dict[str, Dict[str, float]] = {}
    aligned_items = list(aligned.items())
    if not aligned_items:
        raise ValueError("aligned timeframes must not be empty")
    sample_count = len(aligned_items[0][1]["close"])
    agg_cos = cache.buffer("phase_accum_cos", (sample_count,))
    agg_sin = cache.buffer("phase_accum_sin", (sample_count,))
    agg_counts = cache.buffer("phase_accum_counts", (sample_count,), dtype=np.int32)
    agg_cos.fill(0.0)
    agg_sin.fill(0.0)
    agg_counts.fill(0)
    for name, frame in aligned_items:
        close = cache.array(
            f"{name}:close", frame["close"].to_numpy(dtype=np.float32, copy=False)
        )
        returns = cache.buffer(f"{name}:returns", (close.size,))
        returns[0] = np.float32(0.0)
        if close.size > 1:
            np.subtract(close[1:], close[:-1], out=returns[1:])
        phases = compute_phase(
            returns,
            use_float32=True,
            out=cache.buffer(f"{name}:phase", (close.size,)),
        )
        mask = cache.buffer(f"{name}:phase_mask", (close.size,), dtype=bool)
        np.isfinite(phases, out=mask)
        cos_vals = cache.buffer(f"{name}:phase_cos", (close.size,))
        sin_vals = cache.buffer(f"{name}:phase_sin", (close.size,))
        cos_vals.fill(0.0)
        sin_vals.fill(0.0)
        mask_view = mask[: close.size]
        agg_cos_view = agg_cos[: close.size]
        agg_sin_view = agg_sin[: close.size]
        agg_counts_view = agg_counts[: close.size]
        np.cos(phases, out=cos_vals, where=mask_view)
        np.sin(phases, out=sin_vals, where=mask_view)
        valid_count = int(mask_view.sum(dtype=np.int32))
        if valid_count:
            agg_cos_view += cos_vals
            agg_sin_view += sin_vals
            agg_counts_view += mask_view
            local_sum_real = float(np.add.reduce(cos_vals, dtype=np.float64))
            local_sum_imag = float(np.add.reduce(sin_vals, dtype=np.float64))
            local_magnitude = (local_sum_real * local_sum_real + local_sum_imag * local_sum_imag) ** 0.5
            local_kuramoto = float(np.clip(local_magnitude / valid_count, 0.0, 1.0))
        else:
            cos_vals.fill(0.0)
            sin_vals.fill(0.0)
            local_kuramoto = 0.0
        hurst_scratch = cache.buffer(f"{name}:hurst_diff", (close.size,))
        hurst_tau = cache.buffer(f"{name}:hurst_tau", (_DEFAULT_LAGS.size,))
        features[name] = {
            "entropy": _shannon_entropy(returns),
            "hurst": float(
                hurst_exponent(
                    close,
                    use_float32=True,
                    scratch=hurst_scratch,
                    tau_buffer=hurst_tau,
                )
            ),
            "kuramoto": local_kuramoto,
        }
        if book_by_tf and name in book_by_tf:
            book = book_by_tf[name]
            imbalance = book.get("imbalance")
            microprice = book.get("microprice")
            if imbalance is not None:
                features[name]["book_imbalance"] = float(np.nanmean(np.asarray(imbalance, dtype=np.float32)))
            if microprice is not None:
                microprice = np.asarray(microprice, dtype=np.float32)
                price_delta = microprice - close
                features[name]["microprice_basis"] = float(np.nanmean(price_delta))
    valid = agg_counts > 0
    if not np.any(valid):
        phase_coherence = 0.0
    else:
        magnitude = np.hypot(
            agg_cos.astype(np.float64, copy=False),
            agg_sin.astype(np.float64, copy=False),
        )
        counts = agg_counts.astype(np.float64, copy=False)
        coherence = np.divide(
            magnitude,
            counts,
            out=np.zeros_like(magnitude, dtype=np.float64),
            where=valid,
        )
        coherence[~valid] = np.nan
        phase_coherence = float(np.nanmean(coherence))
    benchmark_diff: Dict[str, float] = {}
    if benchmarks:
        flat = _flatten(features)
        for key, expected in benchmarks.items():
            actual = flat.get(key)
            if actual is not None:
                benchmark_diff[key] = float(actual - expected)
    return HierarchicalFeatureResult(
        features=features,
        multi_tf_phase_coherence=phase_coherence,
        benchmarks=benchmark_diff,
    )


__all__ = [
    "FeatureBufferCache",
    "HierarchicalFeatureResult",
    "TimeFrameSpec",
    "compute_hierarchical_features",
]

_DEFAULT_LAGS = np.arange(2, 51, dtype=int)
