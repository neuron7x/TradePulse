# Indicators API Reference

Complete API reference for the TradePulse indicators system. This document covers all public interfaces, protocols, and utilities for building, composing, and executing feature transformations.

---

## Table of Contents

- [Core Interfaces](#core-interfaces)
- [Async Support](#async-support)
- [Observability](#observability)
- [Error Handling](#error-handling)
- [Schema Generation](#schema-generation)
- [Best Practices](#best-practices)

---

## Core Interfaces

### FeatureResult

```python
@dataclass
class FeatureResult:
    """Canonical payload returned by every feature transformer."""
    
    name: str                      # Feature identifier
    value: Any                     # Computed value
    metadata: MetadataDict         # Additional computation metadata
    status: ExecutionStatus        # Success/failed/skipped/partial
    error: Optional[str]           # Error message if failed
    trace_id: str                  # Unique trace ID (UUID)
    timestamp: datetime            # Computation timestamp (UTC)
    provenance: dict[str, Any]     # Audit trail
```

**Methods:**
- `is_success() -> bool`: Check if transformation succeeded
- `is_failed() -> bool`: Check if transformation failed
- `to_model() -> FeatureResultModel`: Convert to validated pydantic model (if available)

**Example:**
```python
result = FeatureResult(
    name="mean_price",
    value=42.5,
    metadata={"window": 20},
    provenance={"version": "1.0", "input_hash": "abc123"}
)

if result.is_success():
    print(f"Computed {result.name}: {result.value}")
```

---

### BaseFeature

```python
class BaseFeature(ABC):
    """Structural contract for every indicator/feature transformer."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        *,
        error_policy: ErrorPolicy = ErrorPolicy.RAISE,
        preprocessor: Optional[PreProcessor] = None,
        postprocessor: Optional[PostProcessor] = None,
    ) -> None: ...
    
    @abstractmethod
    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Produce a feature result from raw input."""
    
    def __call__(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Make feature callable."""
```

**Attributes:**
- `name`: Feature identifier (defaults to class name)
- `error_policy`: How to handle errors (raise/warn/skip/default)
- `preprocessor`: Optional input preprocessor
- `postprocessor`: Optional result postprocessor

**Example:**
```python
class MovingAverage(BaseFeature):
    """Compute moving average over a window."""
    
    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
    
    def transform(self, data: np.ndarray, **kwargs) -> FeatureResult:
        value = np.mean(data[-self.window:])
        return FeatureResult(
            name=self.name,
            value=value,
            metadata={"window": self.window}
        )

# Usage
ma = MovingAverage(window=50, name="ma_50")
result = ma.transform(prices)
```

---

### BaseBlock

```python
class BaseBlock(ABC):
    """Composable container that orchestrates features."""
    
    def __init__(
        self,
        features: Optional[Sequence[BaseFeature]] = None,
        *,
        name: Optional[str] = None,
    ) -> None: ...
    
    @abstractmethod
    def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute the block and return feature mapping."""
```

**Methods:**
- `register(feature: BaseFeature)`: Add a single feature
- `extend(features: Iterable[BaseFeature])`: Add multiple features
- `features -> tuple[BaseFeature, ...]`: Get registered features (immutable)

**Example:**
```python
# Create a block with multiple indicators
block = FeatureBlock([
    KuramotoOrderFeature(name="sync"),
    EntropyFeature(bins=40, name="entropy"),
    HurstFeature(name="hurst")
])

# Execute all features
results = block.run(prices)
# Returns: {"sync": 0.85, "entropy": 2.3, "hurst": 0.6}
```

---

### FunctionalFeature

```python
class FunctionalFeature(BaseFeature):
    """Adapter that wraps a plain function into the feature interface."""
    
    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        metadata: Optional[MetadataDict] = None,
        **kwargs: Any,
    ) -> None: ...
```

**Example:**
```python
# Wrap a simple function
def volatility(data, window=20):
    returns = np.diff(np.log(data))
    return np.std(returns[-window:])

vol_feature = FunctionalFeature(
    volatility,
    name="volatility",
    metadata={"type": "risk_metric"}
)

result = vol_feature.transform(prices, window=30)
```

---

### Protocols

#### PreProcessor

```python
class PreProcessor(Protocol):
    """Protocol for preprocessing feature inputs."""
    
    def process(self, data: FeatureInput, **kwargs: Any) -> FeatureInput:
        """Preprocess input data before transformation."""
```

#### PostProcessor

```python
class PostProcessor(Protocol):
    """Protocol for postprocessing feature results."""
    
    def process(self, result: FeatureResult, **kwargs: Any) -> FeatureResult:
        """Postprocess result after transformation."""
```

**Example:**
```python
class NormalizePreprocessor:
    """Normalize input data to [0, 1]."""
    
    def process(self, data, **kwargs):
        data_min, data_max = np.min(data), np.max(data)
        return (data - data_min) / (data_max - data_min)

feature = KuramotoOrderFeature(
    name="sync_normalized",
    preprocessor=NormalizePreprocessor()
)
```

---

### Enums

#### ErrorPolicy

```python
class ErrorPolicy(str, Enum):
    RAISE = "raise"      # Raise exception immediately
    WARN = "warn"        # Log warning and return error result
    SKIP = "skip"        # Skip silently and return None
    DEFAULT = "default"  # Return default value
```

#### ExecutionStatus

```python
class ExecutionStatus(str, Enum):
    SUCCESS = "success"  # Transformation succeeded
    FAILED = "failed"    # Transformation failed
    SKIPPED = "skipped"  # Transformation skipped
    PARTIAL = "partial"  # Transformation partially succeeded
```

---

## Async Support

### BaseFeatureAsync

```python
class BaseFeatureAsync(ABC):
    """Async version of BaseFeature."""
    
    @abstractmethod
    async def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Produce a feature result asynchronously."""
    
    async def __call__(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Make feature async callable."""
```

**Example:**
```python
class AsyncMarketDataFeature(BaseFeatureAsync):
    """Fetch and process market data asynchronously."""
    
    async def transform(self, symbol: str, **kwargs) -> FeatureResult:
        # Async I/O operation
        data = await fetch_market_data(symbol)
        value = await compute_indicator(data)
        
        return FeatureResult(
            name=self.name,
            value=value,
            metadata={"symbol": symbol}
        )

# Usage
feature = AsyncMarketDataFeature(name="market_indicator")
result = await feature.transform("BTCUSD")
```

---

### FeatureBlockAsync

```python
class FeatureBlockAsync(BaseBlockAsync):
    """Async block that executes features sequentially."""
    
    async def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute all features sequentially with async/await."""
```

---

### FeatureBlockConcurrent

```python
class FeatureBlockConcurrent(BaseBlockAsync):
    """Async block that executes features concurrently."""
    
    async def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute all features concurrently using asyncio.gather."""
```

**Example:**
```python
# Create concurrent block for parallel execution
block = FeatureBlockConcurrent([
    AsyncFeature1(name="f1"),
    AsyncFeature2(name="f2"),
    AsyncFeature3(name="f3"),
])

# All features execute in parallel
results = await block.run(data)
```

---

### AsyncFeatureAdapter

```python
class AsyncFeatureAdapter(BaseFeatureAsync):
    """Adapter to run sync features in async context."""
    
    def __init__(self, sync_feature: BaseFeature, **kwargs) -> None: ...
```

**Example:**
```python
# Wrap sync feature for async pipeline
sync_feature = KuramotoOrderFeature(name="sync")
async_feature = AsyncFeatureAdapter(sync_feature)

# Use in async context
result = await async_feature.transform(data)
```

---

## Observability

### StructuredLogger

```python
class StructuredLogger:
    """Structured JSON logger for feature transformations."""
    
    def __init__(self, name: str, level: int = logging.INFO) -> None: ...
    
    def info(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None: ...
    def warning(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None: ...
    def error(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None: ...
```

**Example:**
```python
from core.indicators.observability import get_logger

logger = get_logger("my_indicators")

logger.info(
    "transform_start",
    trace_id=result.trace_id,
    feature="kuramoto",
    data_size=len(prices)
)
```

---

### IndicatorMetrics

```python
class IndicatorMetrics:
    """Prometheus metrics for indicator transformations."""
    
    @contextmanager
    def measure_transform(self, feature_name: str): ...
    
    def record_success(self, feature_name: str, duration: float) -> None: ...
    def record_error(self, feature_name: str, error_type: str, duration: float) -> None: ...
```

**Metrics Collected:**
- `tradepulse_indicator_transform_latency_seconds`: Transform duration histogram
- `tradepulse_indicator_transform_total`: Total transforms counter (by status)
- `tradepulse_indicator_active_transforms`: Currently active transforms gauge
- `tradepulse_indicator_errors_total`: Total errors counter (by type)

**Example:**
```python
from core.indicators.observability import get_metrics

metrics = get_metrics()

with metrics.measure_transform("entropy"):
    result = entropy_feature.transform(data)
```

---

### with_observability

```python
def with_observability(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[IndicatorMetrics] = None,
) -> Callable: ...
```

**Decorator for automatic logging and metrics:**

```python
from core.indicators.observability import with_observability

class MyFeature(BaseFeature):
    @with_observability()
    def transform(self, data, **kwargs):
        # Automatically logged and measured
        return FeatureResult(name=self.name, value=compute(data))
```

---

## Error Handling

### with_error_handling

```python
def with_error_handling(
    policy: ErrorPolicy = ErrorPolicy.RAISE,
    default_value: Any = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Callable: ...
```

**Example:**
```python
from core.indicators.errors import with_error_handling

class RobustFeature(BaseFeature):
    @with_error_handling(
        policy=ErrorPolicy.DEFAULT,
        default_value=0.0
    )
    def transform(self, data, **kwargs):
        # If this raises, returns 0.0 with PARTIAL status
        return FeatureResult(name=self.name, value=risky_computation(data))
```

---

### CircuitBreaker

```python
class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, threshold: int = 5, timeout: float = 60.0) -> None: ...
    
    def allow_call(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    def reset(self) -> None: ...
```

**Example:**
```python
from core.indicators.errors import CircuitBreaker

breaker = CircuitBreaker(threshold=3, timeout=30)

class ProtectedFeature(BaseFeature):
    @with_error_handling(circuit_breaker=breaker)
    def transform(self, data, **kwargs):
        # Protected by circuit breaker
        return FeatureResult(name=self.name, value=compute(data))
```

---

### ErrorAggregator

```python
class ErrorAggregator:
    """Aggregate errors from batch processing."""
    
    def record(self, feature_name: str, error: Exception) -> None: ...
    def summary(self) -> dict[str, Any]: ...
    def clear(self) -> None: ...
```

**Example:**
```python
from core.indicators.errors import ErrorAggregator

aggregator = ErrorAggregator()

for item in batch:
    try:
        result = feature.transform(item)
    except Exception as e:
        aggregator.record(feature.name, e)

print(aggregator.summary())
# {"total": 5, "by_feature": {...}, "by_type": {...}}
```

---

## Schema Generation

### Generate OpenAPI Spec

```python
from core.indicators.schema import generate_openapi_spec

spec = generate_openapi_spec(
    title="My Indicators API",
    version="1.0.0"
)

# Save to file
import json
with open("openapi.json", "w") as f:
    json.dump(spec, f, indent=2)
```

### Introspect Features

```python
from core.indicators.schema import introspect_feature, introspect_block

# Get feature metadata
metadata = introspect_feature(my_feature)
print(metadata)
# {
#   "name": "my_feature",
#   "class": "MyFeature",
#   "description": "...",
#   "parameters": {...}
# }

# Get block metadata
block_meta = introspect_block(my_block)
```

---

## Best Practices

### 1. Use Type Hints

```python
from typing import Any
import numpy as np

class TypedFeature(BaseFeature):
    def transform(self, data: np.ndarray, window: int = 20, **kwargs: Any) -> FeatureResult:
        ...
```

### 2. Add Comprehensive Docstrings

```python
class WellDocumentedFeature(BaseFeature):
    """Compute well-documented indicator.
    
    This feature calculates... using... algorithm.
    
    Args:
        window: Lookback window size
        method: Computation method ('fast' or 'accurate')
    
    Returns:
        FeatureResult with computed indicator value
    
    Example:
        >>> feature = WellDocumentedFeature(window=50)
        >>> result = feature.transform(prices)
    """
```

### 3. Use Observability

```python
from core.indicators.observability import with_observability

class MonitoredFeature(BaseFeature):
    @with_observability()
    def transform(self, data, **kwargs):
        # Automatically logged and measured
        return FeatureResult(...)
```

### 4. Handle Errors Gracefully

```python
from core.indicators.errors import with_error_handling, ErrorPolicy

class ResilientFeature(BaseFeature):
    @with_error_handling(
        policy=ErrorPolicy.DEFAULT,
        default_value=0.0
    )
    def transform(self, data, **kwargs):
        # Falls back to 0.0 on error
        return FeatureResult(...)
```

### 5. Use Async for I/O

```python
class AsyncIOFeature(BaseFeatureAsync):
    async def transform(self, data, **kwargs):
        # Non-blocking I/O
        external_data = await fetch_data_async()
        return FeatureResult(...)
```

### 6. Compose Features

```python
# Build complex pipelines from simple features
regime_detector = FeatureBlock([
    KuramotoOrderFeature(name="sync"),
    EntropyFeature(name="entropy"),
    HurstFeature(name="hurst"),
])

results = regime_detector.run(prices)
```

---

## Migration Guide

### From Old API to New API

**Old:**
```python
class OldFeature(BaseFeature):
    def transform(self, data, **kwargs):
        return FeatureResult(name=self.name, value=data * 2, metadata={})
```

**New (Fully Compatible):**
```python
class NewFeature(BaseFeature):
    def transform(self, data, **kwargs):
        # All new fields optional - fully backward compatible
        return FeatureResult(
            name=self.name,
            value=data * 2,
            metadata={},
            # New fields auto-populated with defaults:
            # status=ExecutionStatus.SUCCESS,
            # error=None,
            # trace_id=<auto-generated>,
            # timestamp=<auto-generated>,
            # provenance={}
        )
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Integration API](integration-api.md)
- [Monitoring Guide](monitoring.md)
