# SPDX-License-Identifier: MIT
"""Tests for async indicator features."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.indicators.async_base import (
    AsyncFeatureAdapter,
    BaseFeatureAsync,
    FeatureBlockAsync,
    FeatureBlockConcurrent,
)
from core.indicators.base import BaseFeature, FeatureResult


class AsyncDoubleFeature(BaseFeatureAsync):
    """Simple async feature that doubles input."""
    
    async def transform(self, data, **kwargs):
        # Simulate async I/O
        await asyncio.sleep(0.01)
        value = float(data) * 2
        return FeatureResult(
            name=self.name,
            value=value,
            metadata={"async": True},
        )


class SyncTripleFeature(BaseFeature):
    """Sync feature for adapter testing."""
    
    def transform(self, data, **kwargs):
        value = float(data) * 3
        return FeatureResult(name=self.name, value=value)


@pytest.mark.asyncio
async def test_async_feature_basic():
    """Test basic async feature transformation."""
    feature = AsyncDoubleFeature(name="async_double")
    result = await feature.transform(5)
    
    assert result.name == "async_double"
    assert result.value == 10.0
    assert result.metadata["async"] is True
    assert result.is_success()


@pytest.mark.asyncio
async def test_async_feature_callable():
    """Test that async features are callable."""
    feature = AsyncDoubleFeature(name="test")
    result = await feature(10)
    
    assert result.value == 20.0


@pytest.mark.asyncio
async def test_async_block_sequential():
    """Test async block executes features sequentially."""
    block = FeatureBlockAsync([
        AsyncDoubleFeature(name="double"),
        AsyncDoubleFeature(name="double2"),
    ])
    
    results = await block.run(5)
    
    assert results["double"] == 10.0
    assert results["double2"] == 10.0


@pytest.mark.asyncio
async def test_async_block_callable():
    """Test that async blocks are callable."""
    block = FeatureBlockAsync([AsyncDoubleFeature(name="double")])
    results = await block(5)
    
    assert results["double"] == 10.0


@pytest.mark.asyncio
async def test_concurrent_block_faster_than_sequential():
    """Test concurrent block executes features in parallel."""
    import time
    
    # Features that sleep for 0.1 seconds each
    class SlowAsyncFeature(BaseFeatureAsync):
        async def transform(self, data, **kwargs):
            await asyncio.sleep(0.1)
            return FeatureResult(name=self.name, value=data)
    
    # Create 3 features
    features = [SlowAsyncFeature(name=f"f{i}") for i in range(3)]
    
    # Sequential execution
    seq_block = FeatureBlockAsync(features)
    start = time.time()
    await seq_block.run(5)
    seq_duration = time.time() - start
    
    # Concurrent execution
    conc_block = FeatureBlockConcurrent(features)
    start = time.time()
    await conc_block.run(5)
    conc_duration = time.time() - start
    
    # Concurrent should be significantly faster
    # Sequential: ~0.3s (3 x 0.1s), Concurrent: ~0.1s
    assert conc_duration < seq_duration * 0.5


@pytest.mark.asyncio
async def test_async_feature_adapter():
    """Test adapter that wraps sync feature for async use."""
    sync_feature = SyncTripleFeature(name="triple")
    async_feature = AsyncFeatureAdapter(sync_feature)
    
    result = await async_feature.transform(5)
    
    assert result.name == "triple"
    assert result.value == 15.0


@pytest.mark.asyncio
async def test_async_block_register_and_extend():
    """Test registering features in async block."""
    block = FeatureBlockAsync()
    
    block.register(AsyncDoubleFeature(name="f1"))
    block.extend([AsyncDoubleFeature(name="f2"), AsyncDoubleFeature(name="f3")])
    
    assert len(block.features) == 3
    
    results = await block.run(2)
    assert results["f1"] == 4.0
    assert results["f2"] == 4.0
    assert results["f3"] == 4.0


@pytest.mark.asyncio
async def test_concurrent_block_all_features_execute():
    """Test that concurrent block executes all features."""
    features = [AsyncDoubleFeature(name=f"f{i}") for i in range(5)]
    block = FeatureBlockConcurrent(features)
    
    results = await block.run(10)
    
    assert len(results) == 5
    for i in range(5):
        assert results[f"f{i}"] == 20.0


@pytest.mark.asyncio
async def test_async_feature_with_kwargs():
    """Test async feature with additional kwargs."""
    class ConfigurableAsyncFeature(BaseFeatureAsync):
        async def transform(self, data, **kwargs):
            multiplier = kwargs.get("multiplier", 1)
            await asyncio.sleep(0.01)
            return FeatureResult(
                name=self.name,
                value=float(data) * multiplier
            )
    
    feature = ConfigurableAsyncFeature(name="configurable")
    result = await feature.transform(5, multiplier=3)
    
    assert result.value == 15.0


@pytest.mark.asyncio
async def test_async_adapter_preserves_sync_behavior():
    """Test that adapter doesn't change sync feature behavior."""
    sync_feature = SyncTripleFeature(name="sync")
    
    # Direct sync call
    sync_result = sync_feature.transform(10)
    
    # Async via adapter
    async_feature = AsyncFeatureAdapter(sync_feature)
    async_result = await async_feature.transform(10)
    
    assert sync_result.value == async_result.value
    assert sync_result.name == async_result.name


@pytest.mark.asyncio
async def test_concurrent_block_with_mixed_durations():
    """Test concurrent execution with varying durations."""
    class VariableDurationFeature(BaseFeatureAsync):
        def __init__(self, name: str, duration: float):
            super().__init__(name)
            self.duration = duration
        
        async def transform(self, data, **kwargs):
            await asyncio.sleep(self.duration)
            return FeatureResult(name=self.name, value=data * self.duration)
    
    features = [
        VariableDurationFeature("fast", 0.01),
        VariableDurationFeature("medium", 0.05),
        VariableDurationFeature("slow", 0.1),
    ]
    
    block = FeatureBlockConcurrent(features)
    results = await block.run(10)
    
    assert "fast" in results
    assert "medium" in results
    assert "slow" in results


@pytest.mark.asyncio
async def test_async_feature_error_handling():
    """Test error handling in async features."""
    class ErrorAsyncFeature(BaseFeatureAsync):
        async def transform(self, data, **kwargs):
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
    
    feature = ErrorAsyncFeature(name="error")
    
    with pytest.raises(ValueError, match="Async error"):
        await feature.transform(5)
