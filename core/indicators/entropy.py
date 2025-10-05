# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseFeature, FeatureResult


def entropy(series: np.ndarray, bins: int = 30) -> float:
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return 0.0

    finite = np.isfinite(x)
    if not finite.all():
        x = x[finite]
    if x.size == 0:
        return 0.0

    scale = np.max(np.abs(x))
    if scale and np.isfinite(scale):
        x = x / scale

    counts, _ = np.histogram(x, bins=bins, density=False)
    total = counts.sum(dtype=float)
    if total == 0:
        return 0.0

    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def delta_entropy(series: np.ndarray, window: int = 100, bins_range=(10, 50)) -> float:
    """ΔH = H(t) - H(t-τ) using two consecutive windows (last 'window' points)."""
    x = np.asarray(series, dtype=float)
    if x.size < 2 * window:
        return 0.0
    a, b = x[-window * 2 : -window], x[-window:]
    bins = int(np.clip(window // 3, bins_range[0], bins_range[1]))
    return float(entropy(b, bins) - entropy(a, bins))


class EntropyFeature(BaseFeature):
    """Feature wrapper around the Shannon entropy indicator."""

    def __init__(self, bins: int = 30, *, name: str | None = None) -> None:
        super().__init__(name or "entropy")
        self.bins = bins

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        value = entropy(data, bins=self.bins)
        return FeatureResult(name=self.name, value=value, metadata={"bins": self.bins})


class DeltaEntropyFeature(BaseFeature):
    """Feature that computes ΔH over a rolling window."""

    def __init__(
        self,
        window: int = 100,
        bins_range: tuple[int, int] = (10, 50),
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "delta_entropy")
        self.window = window
        self.bins_range = bins_range

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        value = delta_entropy(data, window=self.window, bins_range=self.bins_range)
        metadata = {"window": self.window, "bins_range": self.bins_range}
        return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = [
    "entropy",
    "delta_entropy",
    "EntropyFeature",
    "DeltaEntropyFeature",
]
