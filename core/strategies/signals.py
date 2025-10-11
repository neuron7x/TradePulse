"""Reusable signal generation helpers for CLI integrations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["mean_reversion_signal", "momentum_signal"]


def mean_reversion_signal(
    prices: NDArray[np.float64],
    *,
    lookback: int = 20,
    zscore: float = 1.0,
    risk: float = 1.0,
) -> NDArray[np.float64]:
    """Return a mean-reversion signal clipped to [-1, 1]."""

    series = np.asarray(prices, dtype=float)
    if series.size == 0:
        return np.zeros(0, dtype=float)
    lookback = max(1, int(lookback))
    rolling_mean = np.convolve(series, np.ones(lookback) / lookback, mode="same")
    rolling_var = np.convolve((series - rolling_mean) ** 2, np.ones(lookback) / lookback, mode="same")
    rolling_std = np.sqrt(np.maximum(rolling_var, 1e-12))
    zscores = (series - rolling_mean) / rolling_std
    raw_signal = np.where(zscores > zscore, -1.0, np.where(zscores < -zscore, 1.0, 0.0))
    return np.clip(raw_signal * float(risk), -1.0, 1.0)


def momentum_signal(
    prices: NDArray[np.float64],
    *,
    lookback: int = 10,
    risk: float = 1.0,
) -> NDArray[np.float64]:
    """Return a momentum signal using a moving-average crossover."""

    series = np.asarray(prices, dtype=float)
    if series.size == 0:
        return np.zeros(0, dtype=float)
    short_lb = max(1, min(int(lookback), series.size))
    long_lb = max(short_lb * 2, short_lb + 1)
    short_ma = np.convolve(series, np.ones(short_lb) / short_lb, mode="same")
    long_ma = np.convolve(series, np.ones(long_lb) / long_lb, mode="same")
    raw_signal = np.where(short_ma > long_ma, 1.0, -1.0)
    return np.clip(raw_signal * float(risk), -1.0, 1.0)
