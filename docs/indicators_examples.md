# Indicators Usage Examples

Practical examples and patterns for using the TradePulse indicators API. This guide covers common use cases from simple feature computation to advanced async pipelines.

---

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Creating Custom Features](#creating-custom-features)
3. [Composing Feature Blocks](#composing-feature-blocks)
4. [Async Operations](#async-operations)
5. [Error Handling](#error-handling)
6. [Observability](#observability)
7. [Production Patterns](#production-patterns)

---

## Basic Usage

### Simple Feature Transformation

```python
import numpy as np
from core.indicators.kuramoto import KuramotoOrderFeature

# Create feature
sync_feature = KuramotoOrderFeature(name="market_sync")

# Transform data
prices = np.random.randn(1000)
result = sync_feature.transform(prices)

print(f"Sync order: {result.value}")
print(f"Status: {result.status}")
print(f"Trace ID: {result.trace_id}")
```

### Using Built-in Indicators

```python
from core.indicators.entropy import EntropyFeature
from core.indicators.hurst import HurstFeature
from core.indicators.kuramoto import KuramotoOrderFeature

# Create indicators
entropy = EntropyFeature(bins=40, name="entropy")
hurst = HurstFeature(name="hurst")
sync = KuramotoOrderFeature(name="sync")

# Compute each
prices = load_market_data()
entropy_result = entropy.transform(prices)
hurst_result = hurst.transform(prices)
sync_result = sync.transform(prices)

print(f"Entropy: {entropy_result.value:.3f}")
print(f"Hurst: {hurst_result.value:.3f}")
print(f"Sync: {sync_result.value:.3f}")
```

---

## Creating Custom Features

### Simple Custom Feature

```python
from core.indicators.base import BaseFeature, FeatureResult
import numpy as np

class RSIFeature(BaseFeature):
    """Relative Strength Index indicator."""
    
    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    
    def transform(self, data: np.ndarray, **kwargs) -> FeatureResult:
        # Calculate price changes
        deltas = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        
        # Calculate RSI
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return FeatureResult(
            name=self.name,
            value=rsi,
            metadata={
                "period": self.period,
                "avg_gain": avg_gain,
                "avg_loss": avg_loss,
            },
            provenance={
                "algorithm": "RSI",
                "version": "1.0",
            }
        )

# Usage
rsi = RSIFeature(period=14, name="rsi_14")
result = rsi.transform(prices)
print(f"RSI: {result.value:.2f}")
```

### Functional Feature (Quick Prototyping)

```python
from core.indicators.base import FunctionalFeature
import numpy as np

# Define function
def calculate_volatility(data, window=20):
    returns = np.diff(np.log(data))
    return np.std(returns[-window:]) * np.sqrt(252)  # Annualized

# Wrap as feature
vol_feature = FunctionalFeature(
    calculate_volatility,
    name="volatility",
    metadata={"type": "risk", "annualized": True}
)

# Use like any feature
result = vol_feature.transform(prices, window=30)
print(f"Volatility: {result.value:.2%}")
```

### Feature with Preprocessor

```python
from core.indicators.base import BaseFeature, FeatureResult
import numpy as np

class ZScoreNormalizer:
    """Normalize data using z-score."""
    
    def process(self, data, **kwargs):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

class NormalizedMomentum(BaseFeature):
    """Momentum on normalized data."""
    
    def __init__(self, window: int = 20, **kwargs):
        super().__init__(
            preprocessor=ZScoreNormalizer(),
            **kwargs
        )
        self.window = window
    
    def transform(self, data: np.ndarray, **kwargs) -> FeatureResult:
        # Data is already normalized by preprocessor
        momentum = data[-1] - data[-self.window]
        
        return FeatureResult(
            name=self.name,
            value=momentum,
            metadata={"window": self.window, "normalized": True}
        )

# Usage
feature = NormalizedMomentum(window=20, name="norm_momentum")
result = feature.transform(prices)
```

---

## Composing Feature Blocks

### Basic Block Composition

```python
from core.indicators.base import FeatureBlock
from core.indicators.entropy import EntropyFeature
from core.indicators.hurst import HurstFeature
from core.indicators.kuramoto import KuramotoOrderFeature

# Create a regime detection block
regime_block = FeatureBlock(
    name="regime_detector",
    features=[
        KuramotoOrderFeature(name="sync"),
        EntropyFeature(bins=40, name="entropy"),
        HurstFeature(name="hurst"),
    ]
)

# Execute all features at once
results = regime_block.run(prices)

print("Regime Indicators:")
for name, value in results.items():
    print(f"  {name}: {value:.3f}")
```

### Dynamic Block Construction

```python
from core.indicators.base import FeatureBlock

# Start with empty block
block = FeatureBlock(name="dynamic_block")

# Add features based on config
config = {
    "use_entropy": True,
    "use_hurst": True,
    "use_sync": False,
}

if config["use_entropy"]:
    block.register(EntropyFeature(name="entropy"))

if config["use_hurst"]:
    block.register(HurstFeature(name="hurst"))

if config["use_sync"]:
    block.register(KuramotoOrderFeature(name="sync"))

# Execute
results = block(prices)
```

### Nested Blocks (Fractal Composition)

```python
from core.indicators.base import FeatureBlock

# Low-level features
trend_block = FeatureBlock(
    name="trend",
    features=[
        RSIFeature(period=14, name="rsi"),
        FunctionalFeature(
            lambda x: np.mean(x[-50:]) - np.mean(x[-200:]),
            name="trend_strength"
        ),
    ]
)

# High-level features
volatility_block = FeatureBlock(
    name="volatility",
    features=[
        FunctionalFeature(
            calculate_volatility,
            name="vol_20"
        ),
        HurstFeature(name="hurst"),
    ]
)

# Master block
master_block = FeatureBlock(
    name="market_analysis",
    features=[
        # Note: Blocks can be registered as features!
        # We'd need to create wrapper features, or execute separately
    ]
)

# Execute each block
trend_results = trend_block.run(prices)
vol_results = volatility_block.run(prices)

combined = {**trend_results, **vol_results}
```

---

## Async Operations

### Basic Async Feature

```python
import asyncio
from core.indicators.async_base import BaseFeatureAsync
from core.indicators.base import FeatureResult

class AsyncMarketDataFeature(BaseFeatureAsync):
    """Fetch live market data asynchronously."""
    
    async def transform(self, symbol: str, **kwargs) -> FeatureResult:
        # Simulate async API call
        await asyncio.sleep(0.1)
        
        # Fetch data
        price = await fetch_price(symbol)  # async function
        
        return FeatureResult(
            name=self.name,
            value=price,
            metadata={"symbol": symbol}
        )

# Usage
async def main():
    feature = AsyncMarketDataFeature(name="btc_price")
    result = await feature.transform("BTCUSD")
    print(f"BTC Price: ${result.value:,.2f}")

asyncio.run(main())
```

### Sequential Async Block

```python
from core.indicators.async_base import FeatureBlockAsync

async def main():
    # Features execute one after another (async/await)
    block = FeatureBlockAsync([
        AsyncMarketDataFeature(name="btc"),
        AsyncMarketDataFeature(name="eth"),
        AsyncMarketDataFeature(name="sol"),
    ])
    
    results = await block.run("BTCUSD")
    print(results)

asyncio.run(main())
```

### Concurrent Async Block (Parallel Execution)

```python
from core.indicators.async_base import FeatureBlockConcurrent

async def main():
    # All features execute in parallel!
    block = FeatureBlockConcurrent([
        AsyncMarketDataFeature(name="btc"),
        AsyncMarketDataFeature(name="eth"),
        AsyncMarketDataFeature(name="sol"),
    ])
    
    # Much faster than sequential
    results = await block.run("BTCUSD")
    print(results)

asyncio.run(main())
```

### Mixing Sync and Async

```python
from core.indicators.async_base import AsyncFeatureAdapter
from core.indicators.entropy import EntropyFeature

async def main():
    # Wrap sync feature for async context
    sync_entropy = EntropyFeature(name="entropy")
    async_entropy = AsyncFeatureAdapter(sync_entropy)
    
    # Use in async pipeline
    block = FeatureBlockConcurrent([
        async_entropy,
        AsyncMarketDataFeature(name="data"),
    ])
    
    results = await block.run(prices)

asyncio.run(main())
```

---

## Error Handling

### Using Error Policies

```python
from core.indicators.base import ErrorPolicy
from core.indicators.errors import with_error_handling

class RiskyFeature(BaseFeature):
    """Feature that might fail."""
    
    @with_error_handling(policy=ErrorPolicy.DEFAULT, default_value=0.0)
    def transform(self, data, **kwargs):
        if len(data) < 100:
            raise ValueError("Insufficient data")
        
        # Risky computation
        result = compute_complex_indicator(data)
        
        return FeatureResult(name=self.name, value=result)

# Usage - returns 0.0 if error occurs
feature = RiskyFeature(name="risky")
result = feature.transform(short_data)

if result.status == ExecutionStatus.PARTIAL:
    print(f"Used fallback value: {result.value}")
```

### Circuit Breaker Pattern

```python
from core.indicators.errors import CircuitBreaker, with_error_handling, ErrorPolicy

# Create circuit breaker (opens after 5 failures)
breaker = CircuitBreaker(threshold=5, timeout=60)

class ExternalAPIFeature(BaseFeature):
    """Feature that calls external API."""
    
    @with_error_handling(
        policy=ErrorPolicy.WARN,
        circuit_breaker=breaker
    )
    def transform(self, data, **kwargs):
        # Call external API
        result = call_external_api(data)
        return FeatureResult(name=self.name, value=result)

# Usage
feature = ExternalAPIFeature(name="external")

# First 5 failures recorded
for i in range(10):
    result = feature.transform(data)
    if result.status == ExecutionStatus.SKIPPED:
        print("Circuit breaker is open - calls blocked")
        break
```

### Batch Error Aggregation

```python
from core.indicators.errors import ErrorAggregator

aggregator = ErrorAggregator()

# Process batch
for item in large_batch:
    try:
        result = feature.transform(item)
    except Exception as e:
        aggregator.record(feature.name, e)
        continue

# Get summary
summary = aggregator.summary()
print(f"Total errors: {summary['total']}")
print(f"By feature: {summary['by_feature']}")
print(f"By type: {summary['by_type']}")
```

---

## Observability

### Structured Logging

```python
from core.indicators.observability import get_logger

logger = get_logger("my_indicators")

class LoggedFeature(BaseFeature):
    def transform(self, data, **kwargs):
        trace_id = kwargs.get("trace_id", str(uuid4()))
        
        logger.info(
            "transform_start",
            trace_id=trace_id,
            feature=self.name,
            data_size=len(data)
        )
        
        try:
            result = compute_indicator(data)
            
            logger.info(
                "transform_complete",
                trace_id=trace_id,
                feature=self.name,
                value=result
            )
            
            return FeatureResult(
                name=self.name,
                value=result,
                trace_id=trace_id
            )
        except Exception as e:
            logger.error(
                "transform_error",
                trace_id=trace_id,
                feature=self.name,
                error=str(e)
            )
            raise
```

### Prometheus Metrics

```python
from core.indicators.observability import get_metrics

metrics = get_metrics()

class MeteredFeature(BaseFeature):
    def transform(self, data, **kwargs):
        with metrics.measure_transform(self.name):
            # Automatically measures duration and records metrics
            result = compute_indicator(data)
            return FeatureResult(name=self.name, value=result)
```

### Automatic Observability

```python
from core.indicators.observability import with_observability

class AutoObservedFeature(BaseFeature):
    @with_observability()  # Logs and measures automatically!
    def transform(self, data, **kwargs):
        result = compute_indicator(data)
        return FeatureResult(name=self.name, value=result)
```

---

## Production Patterns

### Complete Production Feature

```python
from core.indicators.base import BaseFeature, FeatureResult, ErrorPolicy
from core.indicators.errors import with_error_handling, CircuitBreaker
from core.indicators.observability import with_observability
import numpy as np

# Shared circuit breaker for all instances
_circuit_breaker = CircuitBreaker(threshold=10, timeout=120)

class ProductionReadyFeature(BaseFeature):
    """Production-ready feature with all best practices."""
    
    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.5,
        **kwargs
    ):
        """Initialize feature.
        
        Args:
            window: Lookback window size
            threshold: Computation threshold
            **kwargs: Additional BaseFeature arguments
        """
        super().__init__(
            error_policy=ErrorPolicy.DEFAULT,
            **kwargs
        )
        self.window = window
        self.threshold = threshold
    
    @with_observability()
    @with_error_handling(
        policy=ErrorPolicy.DEFAULT,
        default_value=0.0,
        circuit_breaker=_circuit_breaker
    )
    def transform(self, data: np.ndarray, **kwargs) -> FeatureResult:
        """Transform data with comprehensive error handling and logging.
        
        Args:
            data: Input price series
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with computed indicator
        """
        # Validate input
        if len(data) < self.window:
            raise ValueError(
                f"Insufficient data: {len(data)} < {self.window}"
            )
        
        # Compute indicator
        windowed_data = data[-self.window:]
        value = np.mean(windowed_data)
        
        # Compute metadata
        metadata = {
            "window": self.window,
            "threshold": self.threshold,
            "data_points": len(data),
        }
        
        # Compute provenance
        provenance = {
            "algorithm": "simple_mean",
            "version": "1.0.0",
            "input_shape": data.shape,
        }
        
        return FeatureResult(
            name=self.name,
            value=value,
            metadata=metadata,
            provenance=provenance,
        )

# Usage
feature = ProductionReadyFeature(
    window=50,
    threshold=0.5,
    name="prod_indicator"
)

result = feature.transform(prices)
```

### Streaming Pipeline

```python
from core.indicators.base import FeatureBlock

class StreamingPipeline:
    """Process streaming data with indicators."""
    
    def __init__(self):
        self.block = FeatureBlock([
            EntropyFeature(bins=40, name="entropy"),
            HurstFeature(name="hurst"),
            RSIFeature(period=14, name="rsi"),
        ])
        
        self.buffer = []
        self.buffer_size = 1000
    
    def process(self, tick: float) -> dict:
        """Process single tick and compute indicators.
        
        Args:
            tick: New price tick
            
        Returns:
            Dictionary of indicator values
        """
        # Add to buffer
        self.buffer.append(tick)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Compute indicators on buffer
        if len(self.buffer) >= 100:  # Minimum data requirement
            data = np.array(self.buffer)
            results = self.block.run(data)
            return results
        else:
            return {}

# Usage
pipeline = StreamingPipeline()

for tick in stream_ticks():
    indicators = pipeline.process(tick)
    if indicators:
        print(f"Indicators: {indicators}")
```

### Multi-Timeframe Analysis

```python
class MultiTimeframeAnalyzer:
    """Analyze indicators across multiple timeframes."""
    
    def __init__(self):
        self.timeframes = {
            "1h": FeatureBlock([
                RSIFeature(period=14, name="rsi_1h"),
                FunctionalFeature(calculate_volatility, name="vol_1h"),
            ]),
            "4h": FeatureBlock([
                RSIFeature(period=14, name="rsi_4h"),
                FunctionalFeature(calculate_volatility, name="vol_4h"),
            ]),
            "1d": FeatureBlock([
                RSIFeature(period=14, name="rsi_1d"),
                FunctionalFeature(calculate_volatility, name="vol_1d"),
            ]),
        }
    
    def analyze(self, data_by_timeframe: dict) -> dict:
        """Analyze all timeframes.
        
        Args:
            data_by_timeframe: Dict mapping timeframe to price data
            
        Returns:
            Nested dict of results by timeframe
        """
        results = {}
        for tf, data in data_by_timeframe.items():
            block = self.timeframes[tf]
            results[tf] = block.run(data)
        return results

# Usage
analyzer = MultiTimeframeAnalyzer()

data = {
    "1h": hourly_prices,
    "4h": four_hour_prices,
    "1d": daily_prices,
}

results = analyzer.analyze(data)
print(results)
# {
#   "1h": {"rsi_1h": 65.2, "vol_1h": 0.023},
#   "4h": {"rsi_4h": 58.1, "vol_4h": 0.031},
#   "1d": {"rsi_1d": 52.3, "vol_1d": 0.042},
# }
```

---

## See Also

- [API Reference](indicators_api.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Integration Guide](integration-api.md)
