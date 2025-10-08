#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Quick demonstration of the enhanced indicators API.

This script showcases:
- Type-safe feature transformations
- Async concurrent execution
- Structured logging and metrics
- Error handling with circuit breaker
- Schema generation

Run with: python examples/indicators_demo.py
"""

import asyncio
import json
import numpy as np

# Core API
from core.indicators.base import (
    BaseFeature,
    ErrorPolicy,
    FeatureBlock,
    FeatureResult,
    FunctionalFeature,
)

# Async support
from core.indicators.async_base import (
    BaseFeatureAsync,
    FeatureBlockConcurrent,
)

# Observability
from core.indicators.observability import (
    get_logger,
    get_metrics,
    with_observability,
)

# Error handling
from core.indicators.errors import (
    CircuitBreaker,
    ErrorAggregator,
    with_error_handling,
)

# Schema generation
from core.indicators.schema import (
    generate_openapi_spec,
    get_feature_result_schema,
    introspect_feature,
)

# Concrete indicators
from core.indicators.entropy import EntropyFeature
from core.indicators.hurst import HurstFeature
from core.indicators.kuramoto import KuramotoOrderFeature


def demo_basic_features():
    """Demonstrate basic feature usage."""
    print("\n=== Basic Feature Usage ===\n")
    
    # Generate sample data
    prices = np.random.randn(1000).cumsum() + 100
    
    # Create features
    entropy = EntropyFeature(bins=40, name="entropy")
    hurst = HurstFeature(name="hurst")
    sync = KuramotoOrderFeature(name="sync")
    
    # Transform
    entropy_result = entropy.transform(prices)
    hurst_result = hurst.transform(prices)
    sync_result = sync.transform(prices)
    
    print(f"Entropy: {entropy_result.value:.3f}")
    print(f"Hurst: {hurst_result.value:.3f}")
    print(f"Sync: {sync_result.value:.3f}")
    print(f"\nTrace ID: {entropy_result.trace_id}")
    print(f"Timestamp: {entropy_result.timestamp}")
    print(f"Status: {entropy_result.status.value}")


def demo_feature_blocks():
    """Demonstrate feature block composition."""
    print("\n=== Feature Block Composition ===\n")
    
    prices = np.random.randn(1000).cumsum() + 100
    
    # Create a block with multiple features
    block = FeatureBlock(
        name="regime_detector",
        features=[
            EntropyFeature(bins=40, name="entropy"),
            HurstFeature(name="hurst"),
            KuramotoOrderFeature(name="sync"),
        ]
    )
    
    # Execute all at once
    results = block.run(prices)
    
    print("Regime Indicators:")
    for name, value in results.items():
        print(f"  {name}: {value:.3f}")


def demo_functional_features():
    """Demonstrate functional feature adapter."""
    print("\n=== Functional Features ===\n")
    
    prices = np.random.randn(1000).cumsum() + 100
    
    # Wrap simple functions
    def volatility(data, window=20):
        returns = np.diff(np.log(data))
        return np.std(returns[-window:]) * np.sqrt(252)
    
    vol_feature = FunctionalFeature(
        volatility,
        name="volatility",
        metadata={"type": "risk", "annualized": True}
    )
    
    result = vol_feature.transform(prices, window=30)
    print(f"Volatility: {result.value:.2%}")
    print(f"Metadata: {dict(result.metadata)}")


async def demo_async_features():
    """Demonstrate async feature execution."""
    print("\n=== Async Feature Execution ===\n")
    
    class AsyncSlowFeature(BaseFeatureAsync):
        """Simulated async I/O feature."""
        
        async def transform(self, data, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API call
            return FeatureResult(
                name=self.name,
                value=np.mean(data),
                metadata={"async": True}
            )
    
    prices = np.random.randn(1000).cumsum() + 100
    
    # Concurrent execution
    block = FeatureBlockConcurrent([
        AsyncSlowFeature(name="feature1"),
        AsyncSlowFeature(name="feature2"),
        AsyncSlowFeature(name="feature3"),
    ])
    
    import time
    start = time.time()
    results = await block.run(prices)
    duration = time.time() - start
    
    print(f"Executed 3 features concurrently in {duration:.3f}s")
    print(f"Results: {list(results.keys())}")


def demo_observability():
    """Demonstrate logging and metrics."""
    print("\n=== Observability ===\n")
    
    logger = get_logger("demo")
    metrics = get_metrics()
    
    prices = np.random.randn(1000).cumsum() + 100
    
    class ObservedFeature(BaseFeature):
        @with_observability()
        def transform(self, data, **kwargs):
            return FeatureResult(
                name=self.name,
                value=np.mean(data),
                metadata={"observed": True}
            )
    
    feature = ObservedFeature(name="observed_mean")
    
    # Automatically logged and measured
    result = feature.transform(prices)
    
    print(f"Result: {result.value:.2f}")
    print("(Logs and metrics automatically recorded)")


def demo_error_handling():
    """Demonstrate error handling patterns."""
    print("\n=== Error Handling ===\n")
    
    breaker = CircuitBreaker(threshold=3)
    
    class RiskyFeature(BaseFeature):
        @with_error_handling(
            policy=ErrorPolicy.DEFAULT,
            default_value=0.0,
            circuit_breaker=breaker
        )
        def transform(self, data, **kwargs):
            if len(data) < 100:
                raise ValueError("Insufficient data")
            return FeatureResult(name=self.name, value=np.mean(data))
    
    feature = RiskyFeature(name="risky")
    
    # Test with short data (will use fallback)
    short_data = np.random.randn(50)
    result = feature.transform(short_data)
    
    print(f"Result: {result.value} (status: {result.status.value})")
    print(f"Error: {result.error}")
    
    # Error aggregator for batch processing
    aggregator = ErrorAggregator()
    
    batch = [np.random.randn(i*10) for i in range(1, 11)]
    for item in batch:
        try:
            result = feature.transform(item)
        except Exception as e:
            aggregator.record(feature.name, e)
    
    summary = aggregator.summary()
    print(f"\nBatch errors: {summary['total']}")


def demo_schema_generation():
    """Demonstrate schema generation."""
    print("\n=== Schema Generation ===\n")
    
    # Get JSON Schema for FeatureResult
    schema = get_feature_result_schema()
    print("FeatureResult JSON Schema:")
    print(json.dumps(schema, indent=2)[:500] + "...")
    
    # Generate OpenAPI spec
    spec = generate_openapi_spec()
    print(f"\nOpenAPI version: {spec['openapi']}")
    print(f"API title: {spec['info']['title']}")
    print(f"Endpoints: {list(spec['paths'].keys())}")
    
    # Introspect feature
    feature = EntropyFeature(bins=40, name="entropy")
    metadata = introspect_feature(feature)
    print(f"\nFeature metadata:")
    print(f"  Name: {metadata['name']}")
    print(f"  Class: {metadata['class']}")
    print(f"  Module: {metadata['module']}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("TradePulse Indicators API Demo")
    print("=" * 60)
    
    demo_basic_features()
    demo_feature_blocks()
    demo_functional_features()
    demo_observability()
    demo_error_handling()
    demo_schema_generation()
    
    print("\n=== Async Demo ===")
    asyncio.run(demo_async_features())
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nFor more examples, see:")
    print("  - docs/indicators_api.md")
    print("  - docs/indicators_examples.md")


if __name__ == "__main__":
    main()
