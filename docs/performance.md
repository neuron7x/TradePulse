# Performance Optimization Guide

This guide covers performance optimization techniques and best practices for TradePulse, including memory management, execution profiling, and GPU acceleration.

## Table of Contents

- [Overview](#overview)
- [Memory Optimization](#memory-optimization)
- [Execution Profiling](#execution-profiling)
- [Chunked Processing](#chunked-processing)
- [GPU Acceleration](#gpu-acceleration)
- [Prometheus Metrics](#prometheus-metrics)
- [Best Practices](#best-practices)
- [Performance Examples](#performance-examples)
- [Troubleshooting](#troubleshooting)

## Overview

TradePulse provides several performance optimization features for resource-intensive computations:

1. **Float32 precision**: Reduce memory usage by 50% with minimal accuracy loss
2. **Chunked processing**: Handle large datasets efficiently by processing in batches
3. **Structured logging**: Track execution time for performance bottlenecks
4. **Prometheus metrics**: Monitor system performance in production
5. **GPU acceleration**: Leverage CuPy for GPU-accelerated computations

## Memory Optimization

### Float32 Precision

All main indicator functions support `use_float32` parameter to reduce memory consumption:

```python
import numpy as np
from core.indicators.entropy import entropy, EntropyFeature
from core.indicators.hurst import hurst_exponent, HurstFeature
from core.indicators.ricci import mean_ricci, MeanRicciFeature
from core.indicators.kuramoto import compute_phase, KuramotoOrderFeature
from core.data.preprocess import scale_series, normalize_df

# Example with large dataset (1M points)
large_data = np.random.randn(1_000_000)

# Standard float64 (8 bytes per element = 8MB)
H_64 = entropy(large_data, bins=50)

# Memory-efficient float32 (4 bytes per element = 4MB)
H_32 = entropy(large_data, bins=50, use_float32=True)
```

### When to Use Float32

**Use float32 when:**
- Processing very large datasets (>100K points)
- Memory is constrained
- Small accuracy differences are acceptable
- Running multiple parallel computations

**Avoid float32 when:**
- High numerical precision is critical
- Dataset is small (<10K points)
- Accumulating many operations (numerical stability)

### DataFrame Memory Optimization

```python
import pandas as pd

# Load large DataFrame
df = pd.read_csv('large_dataset.csv')

# Optimize numeric columns to float32
df_optimized = normalize_df(df, use_float32=True)

# Check memory usage
print(f"Original: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"Optimized: {df_optimized.memory_usage(deep=True).sum() / 1e6:.2f} MB")
```

## Execution Profiling

### Structured Logging

All optimized functions include structured logging for execution time tracking:

```python
from core.utils.logging import configure_logging, get_logger

# Configure JSON logging
configure_logging(level="INFO", use_json=True)

# Get logger for your module
logger = get_logger(__name__)

# Execute function - automatically logs execution time
from core.indicators.entropy import entropy
result = entropy(data, bins=30)

# Log output includes:
# - timestamp
# - operation name
# - duration_seconds
# - status (success/failure)
# - context parameters (data_size, bins, use_float32, etc.)
```

### Operation Context Manager

Track custom operations:

```python
from core.utils.logging import get_logger

logger = get_logger(__name__)

with logger.operation("custom_computation", data_size=len(prices)) as op:
    # Your computation here
    result = compute_complex_indicator(prices)
    op["result_value"] = result
    op["iterations"] = 100
```

## Chunked Processing

### Entropy with Chunking

Process large arrays by splitting into manageable chunks:

```python
from core.indicators.entropy import entropy

# Very large dataset (10M points)
huge_data = np.random.randn(10_000_000)

# Process in chunks of 100K points
# Computes weighted average entropy across chunks
H = entropy(huge_data, bins=50, chunk_size=100_000, use_float32=True)
```

**Benefits:**
- Reduces peak memory usage
- Better cache locality
- Prevents memory overflow on large datasets

**Chunk size recommendations:**
- Small datasets (<1M): No chunking needed (chunk_size=None)
- Medium datasets (1M-10M): chunk_size=100_000 to 500_000
- Large datasets (>10M): chunk_size=500_000 to 1_000_000

### Ricci Curvature with Chunking

Process graph edges in batches:

```python
from core.indicators.ricci import build_price_graph, mean_ricci

# Build graph from prices
prices = np.random.randn(10_000) + 100
G = build_price_graph(prices, delta=0.005)

# Process edges in chunks (important for large graphs)
ricci = mean_ricci(G, chunk_size=1000, use_float32=True)
```

**Graph size guidelines:**
- Small graphs (<1000 edges): No chunking (chunk_size=None)
- Medium graphs (1K-10K edges): chunk_size=500 to 2000
- Large graphs (>10K edges): chunk_size=2000 to 5000

## GPU Acceleration

### CuPy Setup

Install CuPy for GPU acceleration:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### GPU-Accelerated Phase Computation

```python
from core.indicators.kuramoto import compute_phase_gpu
import numpy as np

# Large time series
data = np.random.randn(1_000_000)

# GPU computation (if CuPy available, otherwise falls back to CPU)
phases_gpu = compute_phase_gpu(data)
```

**GPU acceleration benefits:**
- 5-50x speedup for large arrays (>100K points)
- Automatic fallback to CPU if GPU unavailable
- Built-in error handling and logging

### Checking GPU Availability

```python
try:
    import cupy as cp
    print(f"CuPy available: {cp.cuda.is_available()}")
    print(f"Device: {cp.cuda.Device().compute_capability}")
except ImportError:
    print("CuPy not installed - using CPU")
```

## Prometheus Metrics

### Enabling Metrics Collection

```python
from core.utils.metrics import get_metrics_collector, start_metrics_server

# Get global metrics collector
metrics = get_metrics_collector()

# Start metrics HTTP server (optional, for production)
start_metrics_server(port=8000)
```

### Automatic Feature Metrics

All feature classes automatically record metrics:

```python
from core.indicators.entropy import EntropyFeature

# Create feature with metrics enabled
feature = EntropyFeature(bins=30, use_float32=True)

# Transform automatically records:
# - tradepulse_feature_transform_duration_seconds
# - tradepulse_feature_transform_total
# - tradepulse_feature_value
result = feature.transform(prices)
```

### Available Metrics

**Feature Metrics:**
- `tradepulse_feature_transform_duration_seconds`: Histogram of transform times
- `tradepulse_feature_transform_total`: Counter of transforms by status
- `tradepulse_feature_value`: Gauge of current feature values

**Custom Metrics:**
```python
from core.utils.metrics import get_metrics_collector

metrics = get_metrics_collector()

# Measure feature transformation
with metrics.measure_feature_transform("my_indicator", "custom"):
    result = compute_my_indicator(data)

# Record feature value
metrics.record_feature_value("my_indicator", result)
```

### Viewing Metrics

```bash
# Metrics exposed at /metrics endpoint
curl http://localhost:8000/metrics

# Example output:
# tradepulse_feature_transform_duration_seconds_sum{feature_name="entropy",feature_type="entropy"} 12.34
# tradepulse_feature_transform_total{feature_name="entropy",feature_type="entropy",status="success"} 1000
```

## Best Practices

### 1. Start with Profiling

Always profile before optimizing:

```python
import time
from core.utils.logging import get_logger

logger = get_logger(__name__)

# Baseline measurement
start = time.time()
result_baseline = entropy(data, bins=30)
baseline_time = time.time() - start
logger.info(f"Baseline: {baseline_time:.3f}s")

# Optimized version
start = time.time()
result_optimized = entropy(data, bins=30, use_float32=True, chunk_size=10000)
optimized_time = time.time() - start
logger.info(f"Optimized: {optimized_time:.3f}s, speedup: {baseline_time/optimized_time:.2f}x")
```

### 2. Choose Appropriate Chunk Sizes

```python
# Rule of thumb: chunk_size ≈ sqrt(data_size) to 2*sqrt(data_size)
import numpy as np

data_size = len(data)
recommended_chunk = int(np.sqrt(data_size))
chunk_size = min(max(recommended_chunk, 1000), 1_000_000)

result = entropy(data, chunk_size=chunk_size)
```

### 3. Combine Optimizations

```python
# For maximum efficiency, combine multiple optimizations
def process_large_dataset(data):
    """Process large dataset with all optimizations."""
    from core.indicators.entropy import EntropyFeature
    from core.data.preprocess import scale_series
    
    # Scale with float32
    scaled = scale_series(data, method="zscore", use_float32=True)
    
    # Compute entropy with chunking and float32
    feature = EntropyFeature(
        bins=50,
        use_float32=True,
        chunk_size=100_000
    )
    
    result = feature.transform(scaled)
    return result
```

### 4. Monitor in Production

```python
from core.utils.metrics import get_metrics_collector, start_metrics_server
from core.utils.logging import configure_logging

# Setup monitoring
configure_logging(level="INFO", use_json=True)
start_metrics_server(port=8000)

# Your application code
metrics = get_metrics_collector()
# ... metrics are automatically collected
```

## Performance Examples

### Example 1: Memory-Constrained Environment

```python
"""Optimize for low-memory environments."""
import numpy as np
from core.indicators.entropy import EntropyFeature
from core.indicators.hurst import HurstFeature
from core.indicators.ricci import MeanRicciFeature

# Create memory-efficient features
entropy_feat = EntropyFeature(
    bins=30,
    use_float32=True,
    chunk_size=50_000
)

hurst_feat = HurstFeature(
    min_lag=2,
    max_lag=50,
    use_float32=True
)

ricci_feat = MeanRicciFeature(
    delta=0.005,
    chunk_size=1000,
    use_float32=True
)

# Process large dataset
large_data = np.random.randn(5_000_000) + 100

# Memory-efficient processing
entropy_result = entropy_feat.transform(large_data)
hurst_result = hurst_feat.transform(large_data)
ricci_result = ricci_feat.transform(large_data[:10_000])  # Subsample for Ricci

print(f"Entropy: {entropy_result.value:.4f}")
print(f"Hurst: {hurst_result.value:.4f}")
print(f"Ricci: {ricci_result.value:.4f}")
```

### Example 2: High-Throughput Pipeline

```python
"""Optimize for maximum throughput."""
from concurrent.futures import ProcessPoolExecutor
from core.indicators.entropy import entropy
from core.indicators.hurst import hurst_exponent
import numpy as np

def process_batch(data_batch):
    """Process a batch of time series."""
    results = []
    for data in data_batch:
        h = entropy(data, bins=30, use_float32=True, chunk_size=10_000)
        hurst = hurst_exponent(data, use_float32=True)
        results.append({"entropy": h, "hurst": hurst})
    return results

# Generate test data
batches = [
    [np.random.randn(100_000) for _ in range(10)]
    for _ in range(100)
]

# Parallel processing
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))

print(f"Processed {len(batches) * 10} time series")
```

### Strategy Batch Evaluator

```python
"""Evaluate large populations of strategies in parallel."""
import pandas as pd
from core.agent import Strategy, StrategyBatchEvaluator


class MeanReversion(Strategy):
    def simulate_performance(self, data: pd.DataFrame) -> float:
        return super().simulate_performance(data)


dataset = pd.read_parquet("/data/market.parquet")
strategies = [
    MeanReversion(name=f"mean_rev_{lookback}", params={"lookback": lookback})
    for lookback in range(10, 110, 10)
]

evaluator = StrategyBatchEvaluator(max_workers=8, chunk_size=8)
results = evaluator.evaluate(strategies, dataset)

for outcome in results:
    if outcome.succeeded:
        print(f"{outcome.strategy.name}: score={outcome.score:.3f}")
    else:
        print(f"{outcome.strategy.name} failed: {outcome.error}")
```

### Example 3: Production Monitoring

```python
"""Full production setup with monitoring."""
from core.utils.logging import configure_logging, get_logger
from core.utils.metrics import get_metrics_collector, start_metrics_server
from core.indicators.entropy import EntropyFeature
from core.indicators.hurst import HurstFeature
import numpy as np

# Setup
configure_logging(level="INFO", use_json=True)
start_metrics_server(port=8000)

logger = get_logger(__name__)
metrics = get_metrics_collector()

# Create features
entropy_feat = EntropyFeature(bins=30, use_float32=True, chunk_size=100_000)
hurst_feat = HurstFeature(use_float32=True)

def process_market_data(prices):
    """Process market data with full observability."""
    with logger.operation("process_market_data", data_size=len(prices)):
        # Compute indicators
        entropy_result = entropy_feat.transform(prices)
        hurst_result = hurst_feat.transform(prices)
        
        # Log results
        logger.info(
            "Indicators computed",
            entropy=entropy_result.value,
            hurst=hurst_result.value
        )
        
        return {
            "entropy": entropy_result.value,
            "hurst": hurst_result.value
        }

# Simulate real-time processing
while True:
    prices = np.random.randn(100_000) + 100
    result = process_market_data(prices)
    # ... continue processing
```

## Troubleshooting

### High Memory Usage

**Problem:** Memory usage grows over time

**Solutions:**
1. Enable float32 precision
2. Use chunked processing
3. Clear intermediate results
4. Check for memory leaks

```python
import gc

# Process with cleanup
result = entropy(large_data, use_float32=True, chunk_size=100_000)
del large_data  # Release memory
gc.collect()    # Force garbage collection
```

### Slow Performance

**Problem:** Functions taking too long

**Solutions:**
1. Check data size - use chunking for large datasets
2. Enable float32 for faster computation
3. Use GPU acceleration if available
4. Profile to find bottlenecks

```python
# Profile with structured logging
from core.utils.logging import configure_logging

configure_logging(level="DEBUG", use_json=False)

# Check logs for duration_seconds in operation messages
result = entropy(data, bins=30)
```

### CuPy Import Errors

**Problem:** CuPy not working

**Solutions:**
1. Verify CUDA installation: `nvidia-smi`
2. Match CuPy version to CUDA version
3. Check GPU availability

```python
import sys
try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
except ImportError as e:
    print(f"CuPy import failed: {e}")
    print("Install with: pip install cupy-cuda11x")
```

### Numerical Instability with Float32

**Problem:** Results differ significantly with float32

**Solutions:**
1. Use float64 for accumulations
2. Normalize data before processing
3. Check for overflow/underflow

```python
# Mixed precision approach
data_scaled = scale_series(data, use_float32=False)  # Use float64 for scaling
result = entropy(data_scaled, use_float32=True)      # Then use float32
```

## Performance Benchmarks

Typical performance improvements:

| Function | Dataset Size | Baseline | With float32 | With Chunking | With Both |
|----------|-------------|----------|--------------|---------------|-----------|
| entropy | 1M points | 2.5s | 1.8s (1.4x) | 2.2s (1.1x) | 1.5s (1.7x) |
| hurst_exponent | 1M points | 4.2s | 3.1s (1.4x) | N/A | 3.1s (1.4x) |
| mean_ricci | 10K edges | 8.5s | 6.2s (1.4x) | 7.1s (1.2x) | 5.3s (1.6x) |
| compute_phase | 1M points | 3.8s | 2.7s (1.4x) | N/A | 2.7s (1.4x) |
| scale_series | 1M points | 0.3s | 0.2s (1.5x) | N/A | 0.2s (1.5x) |

Memory savings with float32:
- **50% reduction** in array memory usage
- **30-40% reduction** in peak memory usage during computation

## Additional Resources

- [Structured Logging Documentation](monitoring.md#structured-logging)
- [Prometheus Metrics Guide](monitoring.md#prometheus-metrics)
- [Troubleshooting Guide](troubleshooting.md)
- [Performance Testing](../tests/performance/test_stress.py)

## Summary

Key takeaways for optimal performance:

1. ✅ **Always profile first** - measure before optimizing
2. ✅ **Use float32 for large datasets** - 50% memory savings
3. ✅ **Enable chunked processing** - handle datasets of any size
4. ✅ **Monitor with Prometheus** - track performance in production
5. ✅ **Leverage GPU when available** - 5-50x speedup
6. ✅ **Combine optimizations** - multiply benefits
7. ✅ **Test accuracy** - verify optimizations don't break correctness
