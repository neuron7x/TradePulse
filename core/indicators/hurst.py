# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from .base import ArrayLike, BaseFeature

__all__ = ["HurstExponentFeature", "hurst_exponent"]


class HurstExponentFeature(BaseFeature):
    """Estimate the Hurst exponent using log-log regression on lagged ranges."""

    def __init__(self, min_lag: int = 2, max_lag: int = 50) -> None:
        if min_lag <= 0 or max_lag <= 0:
            raise ValueError("lags must be positive integers")
        if min_lag >= max_lag:
            raise ValueError("min_lag must be strictly less than max_lag")
        super().__init__(
            name="hurst_exponent",
            params={"min_lag": int(min_lag), "max_lag": int(max_lag)},
            description="R/S-style Hurst exponent estimated via log-log regression.",
        )
        self._min_lag = int(min_lag)
        self._max_lag = int(max_lag)

    def transform(self, ts: ArrayLike) -> float:
        x = self.coerce_vector(ts)
        if x.size < self._max_lag * 2:
            return 0.5
        lags = np.arange(self._min_lag, self._max_lag + 1)
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        y = np.log(tau)
        X = np.vstack([np.ones_like(lags), np.log(lags)]).T
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        H = beta[1]
        return float(np.clip(H, 0.0, 1.0))


def hurst_exponent(ts: ArrayLike, min_lag: int = 2, max_lag: int = 50) -> float:
    """Functional wrapper for :class:`HurstExponentFeature`."""

    return HurstExponentFeature(min_lag=min_lag, max_lag=max_lag).transform(ts)
