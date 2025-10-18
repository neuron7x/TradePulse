"""Strategy helpers exposed for CLI integrations."""

from .dsl import (
    ComponentParameters,
    ParameterField,
    StrategyComponentConfig,
    StrategyDSLLoader,
    StrategyMetadata,
    StrategyPipeline,
    StrategyPipelineDefinition,
    StrategyPresetRegistry,
    load_strategy_pipeline,
)
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
    "StrategyPipelineDefinition",
    "StrategyPipeline",
    "StrategyDSLLoader",
    "StrategyComponentConfig",
    "StrategyMetadata",
    "ParameterField",
    "ComponentParameters",
    "StrategyPresetRegistry",
    "load_strategy_pipeline",
]
