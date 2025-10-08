# SPDX-License-Identifier: MIT
"""Comprehensive tests for enhanced base indicator functionality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st

from core.indicators.base import (
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureBlock,
    FeatureResult,
    FeatureResultModel,
    FunctionalFeature,
    MetadataDict,
)


class SimpleFeature(BaseFeature):
    """Simple test feature that doubles input."""
    
    def transform(self, data, **kwargs):
        value = float(data) * 2
        return FeatureResult(
            name=self.name,
            value=value,
            metadata={"operation": "double"},
        )


class ErrorFeature(BaseFeature):
    """Feature that always raises an error."""
    
    def transform(self, data, **kwargs):
        raise ValueError("Intentional error for testing")


def test_feature_result_has_all_fields():
    """Test that FeatureResult includes all new fields."""
    result = FeatureResult(name="test", value=42)
    
    assert result.name == "test"
    assert result.value == 42
    assert isinstance(result.metadata, dict)
    assert result.status == ExecutionStatus.SUCCESS
    assert result.error is None
    assert isinstance(result.trace_id, str)
    assert len(result.trace_id) > 0
    assert result.timestamp is not None
    assert isinstance(result.provenance, dict)


def test_feature_result_is_success():
    """Test success status check."""
    result = FeatureResult(
        name="test",
        value=42,
        status=ExecutionStatus.SUCCESS
    )
    assert result.is_success()
    assert not result.is_failed()


def test_feature_result_is_failed():
    """Test failure status check."""
    result = FeatureResult(
        name="test",
        value=None,
        status=ExecutionStatus.FAILED,
        error="Test error"
    )
    assert result.is_failed()
    assert not result.is_success()


@pytest.mark.skipif(
    FeatureResultModel is None,
    reason="pydantic not available"
)
def test_feature_result_to_model():
    """Test conversion to validated pydantic model."""
    result = FeatureResult(name="test", value=42)
    model = result.to_model()
    
    assert model.name == "test"
    assert model.value == 42
    assert model.status == ExecutionStatus.SUCCESS


@pytest.mark.skipif(
    FeatureResultModel is None,
    reason="pydantic not available"
)
def test_feature_result_model_validates_name():
    """Test that model validates non-empty name."""
    with pytest.raises(ValueError, match="Feature name cannot be empty"):
        FeatureResultModel(name="", value=42)
    
    with pytest.raises(ValueError, match="Feature name cannot be empty"):
        FeatureResultModel(name="   ", value=42)


def test_base_feature_with_error_policy():
    """Test feature with error policy configuration."""
    feature = SimpleFeature(
        name="test_feature",
        error_policy=ErrorPolicy.WARN
    )
    
    assert feature.name == "test_feature"
    assert feature.error_policy == ErrorPolicy.WARN


def test_functional_feature_with_metadata():
    """Test FunctionalFeature with static metadata."""
    def compute(data):
        return np.mean(data)
    
    feature = FunctionalFeature(
        compute,
        name="mean",
        metadata={"type": "aggregation", "version": "1.0"}
    )
    
    result = feature.transform(np.array([1, 2, 3, 4, 5]))
    
    assert result.name == "mean"
    assert result.value == 3.0
    assert result.metadata["type"] == "aggregation"
    assert result.metadata["version"] == "1.0"


def test_feature_block_collects_all_results():
    """Test that block collects results from all features."""
    block = FeatureBlock(
        [
            SimpleFeature(name="double"),
            FunctionalFeature(lambda x: x + 10, name="add10"),
        ]
    )
    
    results = block.run(5)
    
    assert results["double"] == 10.0
    assert results["add10"] == 15


def test_feature_block_extend():
    """Test extending block with multiple features."""
    block = FeatureBlock(name="test_block")
    
    features = [
        SimpleFeature(name="f1"),
        SimpleFeature(name="f2"),
    ]
    block.extend(features)
    
    assert len(block.features) == 2
    results = block(3)
    assert results["f1"] == 6.0
    assert results["f2"] == 6.0


def test_feature_result_unique_trace_ids():
    """Test that each result gets a unique trace_id."""
    result1 = FeatureResult(name="test1", value=1)
    result2 = FeatureResult(name="test2", value=2)
    
    assert result1.trace_id != result2.trace_id


def test_feature_provenance_tracking():
    """Test that provenance can be added to results."""
    result = FeatureResult(
        name="test",
        value=42,
        provenance={
            "input_hash": "abc123",
            "version": "1.0",
            "parameters": {"window": 20}
        }
    )
    
    assert result.provenance["input_hash"] == "abc123"
    assert result.provenance["version"] == "1.0"
    assert result.provenance["parameters"]["window"] == 20


# Property-based tests using Hypothesis
@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_simple_feature_always_doubles(value: float):
    """Property test: SimpleFeature always doubles input."""
    feature = SimpleFeature(name="double")
    result = feature.transform(value)
    
    assert abs(result.value - (value * 2)) < 1e-6
    assert result.is_success()


@given(
    st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=100
    )
)
def test_functional_feature_aggregation(values: list[float]):
    """Property test: Functional features correctly aggregate."""
    feature = FunctionalFeature(
        lambda x: np.sum(x),
        name="sum"
    )
    
    result = feature.transform(np.array(values))
    expected_sum = sum(values)
    
    assert abs(result.value - expected_sum) < 1e-3


@given(st.integers(min_value=1, max_value=10))
def test_feature_block_scales_with_features(n_features: int):
    """Property test: Block handles arbitrary number of features."""
    features = [SimpleFeature(name=f"f{i}") for i in range(n_features)]
    block = FeatureBlock(features)
    
    results = block.run(5)
    
    assert len(results) == n_features
    for i in range(n_features):
        assert results[f"f{i}"] == 10.0


def test_error_policy_enum_values():
    """Test all error policy enum values."""
    assert ErrorPolicy.RAISE == "raise"
    assert ErrorPolicy.WARN == "warn"
    assert ErrorPolicy.SKIP == "skip"
    assert ErrorPolicy.DEFAULT == "default"


def test_execution_status_enum_values():
    """Test all execution status enum values."""
    assert ExecutionStatus.SUCCESS == "success"
    assert ExecutionStatus.FAILED == "failed"
    assert ExecutionStatus.SKIPPED == "skipped"
    assert ExecutionStatus.PARTIAL == "partial"


def test_base_feature_default_name():
    """Test that feature uses class name as default."""
    class MyCustomFeature(BaseFeature):
        def transform(self, data, **kwargs):
            return FeatureResult(name=self.name, value=data)
    
    feature = MyCustomFeature()
    assert feature.name == "MyCustomFeature"


def test_base_block_default_name():
    """Test that block uses class name as default."""
    class MyCustomBlock(FeatureBlock):
        pass
    
    block = MyCustomBlock()
    assert block.name == "MyCustomBlock"


def test_feature_callable_interface():
    """Test that features are callable."""
    feature = SimpleFeature(name="test")
    result = feature(10)  # Call directly
    
    assert result.value == 20.0
    assert result.name == "test"


def test_block_callable_interface():
    """Test that blocks are callable."""
    block = FeatureBlock([SimpleFeature(name="double")])
    results = block(5)  # Call directly
    
    assert results["double"] == 10.0


def test_metadata_dict_immutability_concept():
    """Test that metadata is properly handled."""
    metadata: MetadataDict = {"key": "value", "count": 42}
    result = FeatureResult(name="test", value=1, metadata=metadata)
    
    # Verify metadata is accessible
    assert result.metadata["key"] == "value"
    assert result.metadata["count"] == 42
