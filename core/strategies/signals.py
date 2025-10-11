"""Reference signal functions used by CLI smoke tests."""

from __future__ import annotations

import numpy as np


def moving_average_signal(prices: np.ndarray, window: int = 3) -> np.ndarray:
    """Return +1 when price is above its moving average else -1."""

    if window <= 0:
        raise ValueError("window must be positive")
    if prices.size < window:
        raise ValueError("prices length must be >= window")
    rolling = np.convolve(prices, np.ones(window) / window, mode="valid")
    aligned = np.concatenate([np.full(window - 1, rolling[0]), rolling])
    signal = np.where(prices >= aligned, 1.0, -1.0)
    return signal


def threshold_signal(prices: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Simple threshold signal returning 1 when price above threshold."""

    return np.where(prices > threshold, 1.0, -1.0)
