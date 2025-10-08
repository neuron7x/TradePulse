"""Public indicator exports for convenient access in tests and notebooks."""

# Core base classes and types
# Async support
from .async_base import (
    AsyncFeatureAdapter,
    AsyncFeatureTransformer,
    AsyncPostProcessor,
    AsyncPreProcessor,
    BaseBlockAsync,
    BaseFeatureAsync,
    FeatureBlockAsync,
    FeatureBlockConcurrent,
)
from .base import (
    BaseBlock,
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureBlock,
    FeatureInput,
    FeatureResult,
    FeatureResultModel,
    FeatureTransformer,
    FunctionalFeature,
    MetadataDict,
    PostProcessor,
    PreProcessor,
)

# Error handling
from .errors import (
    CircuitBreaker,
    ErrorAggregator,
    ErrorContext,
    ErrorRecoveryConfig,
    create_error_result,
    handle_transform_error,
    with_error_handling,
)

# Concrete indicator implementations
from .kuramoto import (
    KuramotoOrderFeature,
    MultiAssetKuramotoFeature,
    compute_phase,
    compute_phase_gpu,
    kuramoto_order,
    multi_asset_kuramoto,
)
from .kuramoto_ricci_composite import (
    KuramotoRicciComposite,
    MarketPhase,
    TradePulseCompositeEngine,
)
from .multiscale_kuramoto import (
    KuramotoResult,
    MultiScaleKuramoto,
    MultiScaleKuramotoFeature,
    MultiScaleResult,
    TimeFrame,
    WaveletWindowSelector,
)

# Observability
from .observability import (
    OTEL_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    IndicatorMetrics,
    StructuredLogger,
    get_logger,
    get_metrics,
    with_observability,
)

# Schema generation
from .schema import (
    generate_openapi_spec,
    get_block_run_operation_schema,
    get_error_policy_schema,
    get_execution_status_schema,
    get_feature_result_schema,
    get_feature_transform_operation_schema,
    get_pydantic_schema,
    introspect_block,
    introspect_feature,
)
from .temporal_ricci import TemporalRicciAnalyzer

__all__ = [
    # Core base classes
    "BaseFeature",
    "BaseBlock",
    "FeatureBlock",
    "FunctionalFeature",
    "FeatureResult",
    "FeatureResultModel",
    "ErrorPolicy",
    "ExecutionStatus",
    "PreProcessor",
    "PostProcessor",
    "FeatureTransformer",
    "FeatureInput",
    "MetadataDict",
    # Async support
    "AsyncPreProcessor",
    "AsyncPostProcessor",
    "AsyncFeatureTransformer",
    "BaseFeatureAsync",
    "BaseBlockAsync",
    "FeatureBlockAsync",
    "FeatureBlockConcurrent",
    "AsyncFeatureAdapter",
    # Observability
    "StructuredLogger",
    "IndicatorMetrics",
    "get_logger",
    "get_metrics",
    "with_observability",
    "PROMETHEUS_AVAILABLE",
    "OTEL_AVAILABLE",
    # Error handling
    "ErrorContext",
    "ErrorRecoveryConfig",
    "CircuitBreaker",
    "create_error_result",
    "handle_transform_error",
    "with_error_handling",
    "ErrorAggregator",
    # Schema generation
    "get_feature_result_schema",
    "get_pydantic_schema",
    "get_error_policy_schema",
    "get_execution_status_schema",
    "get_feature_transform_operation_schema",
    "get_block_run_operation_schema",
    "generate_openapi_spec",
    "introspect_feature",
    "introspect_block",
    # Concrete implementations
    "compute_phase",
    "compute_phase_gpu",
    "kuramoto_order",
    "multi_asset_kuramoto",
    "KuramotoOrderFeature",
    "MultiAssetKuramotoFeature",
    "KuramotoResult",
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
    "KuramotoRicciComposite",
    "MarketPhase",
    "TradePulseCompositeEngine",
    "TemporalRicciAnalyzer",
]

