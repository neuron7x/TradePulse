from __future__ import annotations

import numpy as np


def deterministic_signal(prices: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Return deterministic trading signals based on price deltas."""

    if prices.size == 0:
        return np.array([], dtype=float)
    deltas = np.diff(prices, prepend=prices[0])
    signals = np.where(deltas > threshold, 1.0, np.where(deltas < -threshold, -1.0, 0.0))
    return signals.astype(float)
