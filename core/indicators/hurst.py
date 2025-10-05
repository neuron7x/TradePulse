# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseFeature, FeatureResult


def hurst_exponent(ts: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
    """Estimate Hurst exponent using R/S-like scaling on differenced series."""
    x = np.asarray(ts, dtype=float)
    if x.size < max_lag * 2:
        return 0.5
    lags = np.arange(min_lag, max_lag + 1)
    tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
    # log-log slope
    y = np.log(tau)
    X = np.vstack([np.ones_like(lags), np.log(lags)]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    H = beta[1]
    return float(np.clip(H, 0.0, 1.0))


class HurstFeature(BaseFeature):
    """Feature wrapper for the Hurst exponent estimate."""

    def __init__(
        self,
        min_lag: int = 2,
        max_lag: int = 50,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name or "hurst_exponent")
        self.min_lag = min_lag
        self.max_lag = max_lag

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        value = hurst_exponent(data, min_lag=self.min_lag, max_lag=self.max_lag)
        metadata = {"min_lag": self.min_lag, "max_lag": self.max_lag}
        return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = ["hurst_exponent", "HurstFeature"]
