"""Strategy helpers exposed for CLI integrations."""

from .objectives import sharpe_ratio
from .signals import moving_average_signal, threshold_signal
from .trading import (
    HurstVPINStrategy,
    KuramotoStrategy,
    TradingStrategy,
    register_strategies,
)

__all__ = [
    "moving_average_signal",
    "threshold_signal",
    "sharpe_ratio",
    "TradingStrategy",
    "KuramotoStrategy",
    "HurstVPINStrategy",
    "register_strategies",
]
