# SPDX-License-Identifier: MIT
"""Entropy-based market uncertainty indicators.

This module provides Shannon entropy and delta entropy calculations for
quantifying market uncertainty and regime changes.

Shannon entropy measures the randomness or unpredictability in price data.
Higher entropy indicates more chaotic or random behavior, while lower entropy
suggests more structured or predictable patterns.

Delta entropy (ΔH) measures the change in entropy over time, which can signal
regime transitions in the market.

References:
    - Shannon, C. E. (1948). A mathematical theory of communication.
      Bell System Technical Journal, 27(3), 379-423.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseFeature, FeatureResult


def entropy(series: np.ndarray, bins: int = 30) -> float:
    """Calculate Shannon entropy of a data series.
    
    Entropy quantifies the uncertainty or randomness in the data distribution.
    The series is normalized and binned, then Shannon entropy is computed as:
    
        H = -Σ p(i) * log(p(i))
    
    where p(i) is the probability of bin i.
    
    Args:
        series: 1D array of numeric data (typically prices or returns)
        bins: Number of bins for histogram discretization (default: 30)
        
    Returns:
        Shannon entropy value. Higher values indicate more randomness/chaos.
        Returns 0.0 for empty or invalid input.
        
    Example:
        >>> prices = np.array([100, 101, 102, 101, 100, 99, 100, 101])
        >>> H = entropy(prices, bins=10)
        >>> print(f"Entropy: {H:.3f}")
        Entropy: 1.234
        
    Note:
        - Data is automatically scaled to [-1, 1] range for numerical stability
        - Invalid values (NaN, inf) are filtered out
        - Returns 0.0 if no valid data remains after filtering
    """
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return 0.0

    # Filter out non-finite values
    finite = np.isfinite(x)
    if not finite.all():
        x = x[finite]
    if x.size == 0:
        return 0.0

    # Normalize to [-1, 1] for numerical stability
    scale = np.max(np.abs(x))
    if scale and np.isfinite(scale):
        x = x / scale

    # Compute histogram
    counts, _ = np.histogram(x, bins=bins, density=False)
    total = counts.sum(dtype=float)
    if total == 0:
        return 0.0

    # Calculate Shannon entropy
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def delta_entropy(series: np.ndarray, window: int = 100, bins_range=(10, 50)) -> float:
    """Calculate delta entropy (change in entropy over time).
    
    Delta entropy (ΔH) measures the rate of change in market uncertainty by
    comparing entropy between two consecutive time windows:
    
        ΔH = H(t) - H(t-τ)
    
    where τ is the window size. Positive ΔH indicates increasing chaos,
    negative ΔH suggests decreasing uncertainty.
    
    Args:
        series: 1D array of numeric data (typically prices)
        window: Size of each time window for entropy calculation (default: 100)
        bins_range: (min_bins, max_bins) for adaptive histogram binning.
                   Actual bins = clip(window // 3, min_bins, max_bins)
                   Default: (10, 50)
        
    Returns:
        Delta entropy value. Positive values indicate increasing uncertainty,
        negative values indicate decreasing uncertainty. Returns 0.0 if
        insufficient data (need at least 2 * window points).
        
    Example:
        >>> prices = np.linspace(100, 110, 300)  # Trending market
        >>> dH = delta_entropy(prices, window=100)
        >>> if dH > 0.5:
        ...     print("Market becoming more chaotic")
        ... elif dH < -0.5:
        ...     print("Market becoming more structured")
        
    Note:
        - Requires at least 2 * window data points
        - Bins are adaptively chosen based on window size
        - Useful for detecting regime transitions
    """
    x = np.asarray(series, dtype=float)
    if x.size < 2 * window:
        return 0.0
    
    # Split into two consecutive windows
    a, b = x[-window * 2 : -window], x[-window:]
    
    # Adaptive bin selection
    bins = int(np.clip(window // 3, bins_range[0], bins_range[1]))
    
    # Compute entropy difference
    return float(entropy(b, bins) - entropy(a, bins))


class EntropyFeature(BaseFeature):
    """Feature wrapper for Shannon entropy indicator.
    
    This class wraps the entropy() function as a BaseFeature, making it
    compatible with the TradePulse feature pipeline and composition system.
    
    Attributes:
        bins: Number of histogram bins for entropy calculation
        name: Feature identifier
        
    Example:
        >>> from core.indicators.entropy import EntropyFeature
        >>> import numpy as np
        >>> 
        >>> feature = EntropyFeature(bins=50, name="market_entropy")
        >>> prices = np.random.randn(200) * 10 + 100
        >>> result = feature.transform(prices)
        >>> 
        >>> print(f"{result.name}: {result.value:.3f}")
        >>> print(f"Metadata: {result.metadata}")
        market_entropy: 2.345
        Metadata: {'bins': 50}
    """

    def __init__(self, bins: int = 30, *, name: str | None = None) -> None:
        """Initialize entropy feature.
        
        Args:
            bins: Number of bins for histogram discretization (default: 30)
            name: Optional custom name for the feature (default: "entropy")
        """
        super().__init__(name or "entropy")
        self.bins = bins

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute Shannon entropy of input data.
        
        Args:
            data: 1D array of numeric values (typically prices)
            **_: Additional keyword arguments (ignored)
            
        Returns:
            FeatureResult containing entropy value and metadata
        """
        value = entropy(data, bins=self.bins)
        return FeatureResult(name=self.name, value=value, metadata={"bins": self.bins})


class DeltaEntropyFeature(BaseFeature):
    """Feature wrapper for delta entropy (rate of entropy change).
    
    This feature computes the change in Shannon entropy over time by comparing
    entropy between consecutive time windows. Useful for detecting regime
    transitions and changes in market dynamics.
    
    Attributes:
        window: Size of each time window
        bins_range: (min, max) range for adaptive bin selection
        name: Feature identifier
        
    Example:
        >>> from core.indicators.entropy import DeltaEntropyFeature
        >>> import numpy as np
        >>> 
        >>> feature = DeltaEntropyFeature(window=100, bins_range=(10, 50))
        >>> prices = np.linspace(100, 110, 300)  # Trending market
        >>> result = feature.transform(prices)
        >>> 
        >>> if result.value > 0.5:
        ...     print("Market becoming more chaotic")
        ... elif result.value < -0.5:
        ...     print("Market becoming more structured")
    """

    def __init__(
        self,
        window: int = 100,
        bins_range: tuple[int, int] = (10, 50),
        *,
        name: str | None = None,
    ) -> None:
        """Initialize delta entropy feature.
        
        Args:
            window: Size of each time window (default: 100)
            bins_range: (min_bins, max_bins) for histogram (default: (10, 50))
            name: Optional custom name (default: "delta_entropy")
        """
        super().__init__(name or "delta_entropy")
        self.window = window
        self.bins_range = bins_range

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute delta entropy (ΔH) of input data.
        
        Args:
            data: 1D array of numeric values (typically prices)
            **_: Additional keyword arguments (ignored)
            
        Returns:
            FeatureResult containing ΔH value and metadata
            
        Raises:
            ValueError: If data has fewer than 2 * window points
        """
        value = delta_entropy(data, window=self.window, bins_range=self.bins_range)
        metadata = {"window": self.window, "bins_range": self.bins_range}
        return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = [
    "entropy",
    "delta_entropy",
    "EntropyFeature",
    "DeltaEntropyFeature",
]
