"""Hierarchical feature computation for multi-timeframe analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.data.resampling import align_timeframes
from .entropy import entropy
from .hurst import hurst_exponent
from .kuramoto import compute_phase, kuramoto_order


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
        arr = np.asarray(values, dtype=np.float32)
        self.store[key] = arr
        return arr


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
    phase_stack = []
    for name, frame in aligned.items():
        close = cache.array(f"{name}:close", frame["close"].to_numpy())
        returns = np.diff(close, prepend=close[0])
        phases = compute_phase(returns)
        phase_stack.append(phases)
        features[name] = {
            "entropy": float(entropy(returns, use_float32=True)),
            "hurst": float(hurst_exponent(close, use_float32=True)),
            "kuramoto": float(kuramoto_order(phases)),
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
    phase_matrix = np.vstack(phase_stack)
    phase_coherence = float(np.nanmean(kuramoto_order(phase_matrix)))
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

