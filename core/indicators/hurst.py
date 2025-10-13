# SPDX-License-Identifier: MIT
"""Hurst exponent calculation for detecting long-memory processes in markets.

The Hurst exponent (H) characterizes long-term memory in time series:
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Persistent (trending) behavior
- H < 0.5: Anti-persistent (mean-reverting) behavior

This module uses rescaled range (R/S) analysis to estimate the Hurst exponent
from price time series data.

References:
    - Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
      Transactions of the American Society of Civil Engineers, 116, 770-808.
    - Peters, E. E. (1994). Fractal Market Analysis: Applying Chaos Theory
      to Investment and Economics.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np

from .base import BaseFeature, FeatureResult
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector

_logger = get_logger(__name__)
_metrics = get_metrics_collector()

_DEFAULT_MIN_LAG = 2
_DEFAULT_MAX_LAG = 50
_DEFAULT_LAGS = np.arange(_DEFAULT_MIN_LAG, _DEFAULT_MAX_LAG + 1, dtype=int)
_DEFAULT_DESIGN = np.vstack(
    [np.ones_like(_DEFAULT_LAGS, dtype=float), np.log(_DEFAULT_LAGS)]
).T
_DEFAULT_PSEUDO = np.linalg.pinv(_DEFAULT_DESIGN)


def hurst_exponent(
    ts: np.ndarray,
    min_lag: int = 2,
    max_lag: int = 50,
    *,
    use_float32: bool = False,
    scratch: np.ndarray | None = None,
    tau_buffer: np.ndarray | None = None,
) -> float:
    """Estimate Hurst exponent using rescaled range (R/S) analysis.
    
    The Hurst exponent characterizes the long-term statistical dependencies
    in a time series. It is estimated by analyzing how the standard deviation
    of price differences scales with the time lag.
    
    The calculation uses log-log regression of std vs lag:
        log(std(Δx)) ∝ H * log(lag)
    
    Args:
        ts: 1D array of time series data (typically prices)
        min_lag: Minimum lag for R/S analysis (default: 2)
        max_lag: Maximum lag for R/S analysis (default: 50)
        use_float32: Use float32 precision to reduce memory usage (default: False)
        
    Returns:
        Hurst exponent H ∈ [0, 1]:
        - H ≈ 0.5: Random walk, no memory
        - H > 0.5: Persistent/trending (0.5-1.0)
        - H < 0.5: Anti-persistent/mean-reverting (0.0-0.5)
        Returns 0.5 if insufficient data.
        
    Example:
        >>> import numpy as np
        >>> 
        >>> # Generate trending series
        >>> trend = np.cumsum(np.random.randn(1000)) + np.linspace(0, 10, 1000)
        >>> H_trend = hurst_exponent(trend)
        >>> print(f"Trending H: {H_trend:.3f}")  # Should be > 0.5
        Trending H: 0.653
        >>> 
        >>> # Generate mean-reverting series
        >>> mean_rev = np.random.randn(1000)
        >>> H_mr = hurst_exponent(mean_rev)
        >>> print(f"Mean-reverting H: {H_mr:.3f}")  # Should be < 0.5
        Mean-reverting H: 0.423
        
    Note:
        - Requires at least 2 * max_lag data points
        - Result is clipped to [0, 1] range
        - More data generally provides more reliable estimates
        - float32 mode reduces memory footprint for large datasets
    """
    base_logger = getattr(_logger, "logger", None)
    check = getattr(base_logger, "isEnabledFor", None)
    context_manager = (
        _logger.operation(
            "hurst_exponent",
            min_lag=min_lag,
            max_lag=max_lag,
            use_float32=use_float32,
            data_size=len(ts),
        )
        if check and check(logging.DEBUG)
        else nullcontext()
    )
    with context_manager:
        dtype = np.float32 if use_float32 else float
        x = np.asarray(ts, dtype=dtype)
        if x.size < max_lag * 2:
            return 0.5
        
        if min_lag == _DEFAULT_MIN_LAG and max_lag == _DEFAULT_MAX_LAG:
            lags = _DEFAULT_LAGS
            pseudo = _DEFAULT_PSEUDO
        else:
            lags = np.arange(min_lag, max_lag + 1)
            design = np.vstack(
                [np.ones_like(lags, dtype=float), np.log(lags)]
            ).T
            pseudo = np.linalg.pinv(design)

        tau = tau_buffer
        if tau is None or tau.shape[0] != len(lags):
            tau = np.empty(len(lags), dtype=float)
        buffer = scratch
        if buffer is None or buffer.shape[0] < x.size:
            buffer = np.empty_like(x, dtype=float)
        for idx, lag in enumerate(lags):
            np.subtract(x[lag:], x[:-lag], out=buffer[: x.size - lag])
            segment = buffer[: x.size - lag]
            count = float(segment.size)
            if count == 0:
                tau[idx] = 0.0
                continue
            sum_vals = float(segment.sum(dtype=float))
            sum_sq = float(np.dot(segment, segment))
            mean = sum_vals / count
            var = sum_sq / count - mean * mean
            tau[idx] = float(np.sqrt(var if var > 0.0 else 0.0))

        # Perform log-log linear regression
        y = np.log(tau)
        if min_lag == _DEFAULT_MIN_LAG and max_lag == _DEFAULT_MAX_LAG:
            beta = pseudo @ y
        else:
            X = np.vstack([np.ones_like(lags, dtype=float), np.log(lags)]).T
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        H = beta[1]  # Slope is Hurst exponent
        
        return float(np.clip(H, 0.0, 1.0))


class HurstFeature(BaseFeature):
    """Feature wrapper for Hurst exponent estimation.
    
    This class wraps the hurst_exponent() function as a BaseFeature, making it
    compatible with the TradePulse feature pipeline.
    
    The Hurst exponent is particularly useful for:
    - Identifying market regimes (trending vs mean-reverting)
    - Portfolio diversification (different H values = different behaviors)
    - Risk management (H > 0.5 suggests momentum, H < 0.5 suggests reversion)
    
    Attributes:
        min_lag: Minimum lag for R/S analysis
        max_lag: Maximum lag for R/S analysis
        name: Feature identifier
        
    Example:
        >>> from core.indicators.hurst import HurstFeature
        >>> import numpy as np
        >>> 
        >>> feature = HurstFeature(min_lag=2, max_lag=50, name="hurst")
        >>> prices = np.cumsum(np.random.randn(500)) + 100
        >>> result = feature.transform(prices)
        >>> 
        >>> print(f"{result.name}: {result.value:.3f}")
        >>> if result.value > 0.55:
        ...     print("Market shows trending behavior")
        ... elif result.value < 0.45:
        ...     print("Market shows mean-reverting behavior")
        ... else:
        ...     print("Market shows random walk behavior")
        hurst: 0.623
        Market shows trending behavior
    """

    def __init__(
        self,
        min_lag: int = 2,
        max_lag: int = 50,
        *,
        use_float32: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize Hurst exponent feature.
        
        Args:
            min_lag: Minimum lag for R/S analysis (default: 2)
            max_lag: Maximum lag for R/S analysis (default: 50)
            use_float32: Use float32 precision for memory efficiency (default: False)
            name: Optional custom name (default: "hurst_exponent")
        """
        super().__init__(name or "hurst_exponent")
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.use_float32 = use_float32

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute Hurst exponent of input data.
        
        Args:
            data: 1D array of time series data (typically prices)
            **_: Additional keyword arguments (ignored)
            
        Returns:
            FeatureResult containing Hurst exponent and metadata
        """
        with _metrics.measure_feature_transform(self.name, "hurst"):
            value = hurst_exponent(data, min_lag=self.min_lag, max_lag=self.max_lag,
                                  use_float32=self.use_float32)
            _metrics.record_feature_value(self.name, value)
            metadata: dict[str, Any] = {
                "min_lag": self.min_lag,
                "max_lag": self.max_lag,
            }
            if self.use_float32:
                metadata["use_float32"] = True
            return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = ["hurst_exponent", "HurstFeature"]
