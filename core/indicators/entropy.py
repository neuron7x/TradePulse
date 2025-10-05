# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import ArrayLike, BaseFeature

__all__ = [
    "EntropyFeature",
    "DeltaEntropyFeature",
    "entropy",
    "delta_entropy",
]


class EntropyFeature(BaseFeature):
    """Shannon entropy computed from histogrammed price/feature values."""

    def __init__(self, bins: int = 30) -> None:
        if bins <= 0:
            raise ValueError("EntropyFeature requires a positive number of bins")
        super().__init__(
            name="entropy",
            params={"bins": int(bins)},
            description="Shannon entropy computed over a fixed histogram.",
        )
        self._bins = int(bins)

    def transform(self, series: ArrayLike) -> float:
        x = self.coerce_vector(series)
        if x.size == 0:
            return 0.0
        counts, _ = np.histogram(x, bins=self._bins, density=True)
        p = counts[counts > 0]
        if p.size == 0:
            return 0.0
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())


class DeltaEntropyFeature(BaseFeature):
    """Entropy difference between consecutive rolling windows."""

    def __init__(
        self,
        window: int = 100,
        bins_range: Sequence[int] = (10, 50),
    ) -> None:
        if window <= 0:
            raise ValueError("DeltaEntropyFeature requires a positive window")
        if len(bins_range) != 2:
            raise ValueError("bins_range must be a pair (min_bins, max_bins)")
        lo, hi = int(bins_range[0]), int(bins_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("bins_range must contain positive integers")
        if lo > hi:
            raise ValueError("bins_range minimum cannot exceed maximum")
        super().__init__(
            name="delta_entropy",
            params={"window": int(window), "bins_range": (lo, hi)},
            description=(
                "Difference between the entropy of the latest window and the "
                "preceding window of equal length."
            ),
        )
        self._window = int(window)
        self._bins_range = (lo, hi)

    def transform(self, series: ArrayLike) -> float:
        x = self.coerce_vector(series)
        if x.size < 2 * self._window:
            return 0.0
        a = x[-2 * self._window : -self._window]
        b = x[-self._window :]
        bins = int(np.clip(self._window // 3, self._bins_range[0], self._bins_range[1]))
        entropy_feature = EntropyFeature(bins=bins)
        return float(entropy_feature.transform(b) - entropy_feature.transform(a))


def entropy(series: ArrayLike, bins: int = 30) -> float:
    """Functional wrapper for :class:`EntropyFeature`."""

    return EntropyFeature(bins=bins).transform(series)


def delta_entropy(
    series: ArrayLike,
    window: int = 100,
    bins_range: Sequence[int] = (10, 50),
) -> float:
    """Functional wrapper for :class:`DeltaEntropyFeature`."""

    return DeltaEntropyFeature(window=window, bins_range=bins_range).transform(series)
