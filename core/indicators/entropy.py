# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np

def entropy(series: np.ndarray, bins: int = 30) -> float:
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=bins, density=True)
    p = counts[counts > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def delta_entropy(series: np.ndarray, window: int = 100, bins_range=(10,50)) -> float:
    """ΔH = H(t) - H(t-τ) using two consecutive windows (last 'window' points)."""
    x = np.asarray(series, dtype=float)
    if x.size < 2*window:
        return 0.0
    a, b = x[-window*2:-window], x[-window:]
    bins = int(np.clip(window//3, bins_range[0], bins_range[1]))
    return float(entropy(b, bins) - entropy(a, bins))
