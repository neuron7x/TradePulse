"""Lightweight indicator helpers powering portfolio strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .hurst import hurst_exponent
from .kuramoto import compute_phase


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Input series must be one-dimensional")
    return array


def _fill_missing(series: np.ndarray) -> np.ndarray:
    if series.size == 0:
        return series
    mask = np.isfinite(series)
    if not mask.any():
        return np.zeros_like(series)
    if mask.all():
        return series
    idx = np.flatnonzero(mask)
    filled = np.interp(np.arange(series.size), idx, series[mask])
    return filled


@dataclass(slots=True)
class KuramotoIndicator:
    """Compute rolling Kuramoto-style synchronisation scores."""

    window: int = 200
    coupling: float = 1.0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.coupling <= 0:
            raise ValueError("coupling must be positive")

    def compute(self, prices: Iterable[float]) -> np.ndarray:
        raw = _as_float_array(prices)
        if raw.size == 0:
            return np.empty(0, dtype=float)
        if not np.isfinite(raw).any():
            return np.zeros_like(raw, dtype=float)
        series = _fill_missing(raw)
        phases = compute_phase(series)
        complex_phase = np.exp(1j * phases)
        cumulative = np.cumsum(complex_phase)
        result = np.zeros_like(series, dtype=float)
        min_samples = min(self.window, 10)
        for idx in range(series.size):
            start = max(0, idx - self.window + 1)
            count = idx - start + 1
            if count < min_samples:
                continue
            total = cumulative[idx] - (cumulative[start - 1] if start > 0 else 0.0)
            order = np.abs(total) / float(count)
            scaled = 2.0 * (order - 0.5)
            result[idx] = float(np.clip(self.coupling * scaled, -1.0, 1.0))
        return result


@dataclass(slots=True)
class HurstIndicator:
    """Rolling Hurst exponent estimator."""

    window: int = 100
    min_lag: int = 2
    max_lag: int | None = None

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.min_lag <= 0:
            raise ValueError("min_lag must be positive")
        if self.max_lag is not None and self.max_lag <= self.min_lag:
            raise ValueError("max_lag must exceed min_lag")

    def compute(self, prices: Iterable[float]) -> np.ndarray:
        series = _as_float_array(prices)
        if series.size == 0:
            return np.empty(0, dtype=float)
        series = _fill_missing(series)
        result = np.full(series.size, 0.5, dtype=float)
        for idx in range(series.size):
            start = max(0, idx - self.window + 1)
            window_slice = series[start : idx + 1]
            available = window_slice.size // 2
            if available <= self.min_lag:
                continue
            max_lag = available
            if self.max_lag is not None:
                max_lag = min(max_lag, self.max_lag)
            value = hurst_exponent(window_slice, min_lag=self.min_lag, max_lag=max_lag)
            result[idx] = float(np.clip(value, 0.0, 1.0))
        return result


@dataclass(slots=True)
class VPINIndicator:
    """Volume-synchronised probability of informed trading."""

    bucket_size: int = 50
    threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")

    def compute(self, volume_data: Iterable[Iterable[float]]) -> np.ndarray:
        array = np.asarray(volume_data, dtype=float)
        if array.size == 0:
            return np.empty(0, dtype=float)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError("volume_data must have columns [volume, buy_volume, sell_volume]")
        total = np.clip(np.nan_to_num(array[:, 0], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        buy = np.clip(np.nan_to_num(array[:, 1], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        sell = np.clip(np.nan_to_num(array[:, 2], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        imbalance = np.abs(buy - sell)
        cum_total = np.cumsum(total)
        cum_imbalance = np.cumsum(imbalance)
        result = np.zeros(total.size, dtype=float)
        for idx in range(total.size):
            start = max(0, idx - self.bucket_size + 1)
            total_sum = cum_total[idx] - (cum_total[start - 1] if start > 0 else 0.0)
            if total_sum <= 0.0:
                result[idx] = 0.0
                continue
            imb_sum = cum_imbalance[idx] - (cum_imbalance[start - 1] if start > 0 else 0.0)
            result[idx] = float(np.clip(imb_sum / total_sum, 0.0, 1.0))
        return result
