"""Registry of reference strategies."""

from strategies.amm import AMMConfig, AMMStrategy
from strategies.meanrev import MeanReversionConfig, MeanReversionStrategy
from strategies.trend import TrendStrategy, TrendStrategyConfig

__all__ = [
    "TrendStrategy",
    "TrendStrategyConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "AMMStrategy",
    "AMMConfig",
]

