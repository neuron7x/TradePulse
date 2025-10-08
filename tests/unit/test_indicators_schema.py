# SPDX-License-Identifier: MIT
"""Tests for schema generation."""

from __future__ import annotations

import json

import pytest

from core.indicators.base import (
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureBlock,
    FeatureResult,
)
from core.indicators.schema import (
    generate_openapi_spec,
    get_error_policy_schema,
    get_execution_status_schema,
    get_feature_result_schema,
    introspect_block,
    introspect_feature,
)


class SimpleFeature(BaseFeature):
    """A simple test feature for schema generation."""
    
    def transform(self, data, window: int = 10, **kwargs):
        """Transform with a window parameter."""
        return FeatureResult(name=self.name, value=float(data) * 2)


def test_feature_result_schema_structure():
    """Test that FeatureResult schema has correct structure."""
    schema = get_feature_result_schema()
    
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "value" in schema["properties"]
    assert "metadata" in schema["properties"]
    assert "status" in schema["properties"]
    assert "error" in schema["properties"]
    assert "trace_id" in schema["properties"]
    assert "timestamp" in schema["properties"]
    assert "provenance" in schema["properties"]


def test_feature_result_schema_required_fields():
    """Test that required fields are marked correctly."""
    schema = get_feature_result_schema()
    
    assert "required" in schema
    assert "name" in schema["required"]
    assert "value" in schema["required"]
    assert "status" in schema["required"]


def test_error_policy_schema():
    """Test ErrorPolicy schema generation."""
    schema = get_error_policy_schema()
    
    assert schema["type"] == "string"
    assert "enum" in schema
    assert "raise" in schema["enum"]
    assert "warn" in schema["enum"]
    assert "skip" in schema["enum"]
    assert "default" in schema["enum"]


def test_execution_status_schema():
    """Test ExecutionStatus schema generation."""
    schema = get_execution_status_schema()
    
    assert schema["type"] == "string"
    assert "enum" in schema
    assert "success" in schema["enum"]
    assert "failed" in schema["enum"]
    assert "skipped" in schema["enum"]
    assert "partial" in schema["enum"]


def test_openapi_spec_structure():
    """Test OpenAPI spec has correct structure."""
    spec = generate_openapi_spec()
    
    assert spec["openapi"] == "3.0.3"
    assert "info" in spec
    assert "paths" in spec
    assert "components" in spec


def test_openapi_spec_info():
    """Test OpenAPI spec info section."""
    spec = generate_openapi_spec(
        title="Test API",
        version="2.0.0",
        description="Test description"
    )
    
    assert spec["info"]["title"] == "Test API"
    assert spec["info"]["version"] == "2.0.0"
    assert spec["info"]["description"] == "Test description"


def test_openapi_spec_paths():
    """Test OpenAPI spec has correct paths."""
    spec = generate_openapi_spec()
    
    assert "/features/{feature_name}/transform" in spec["paths"]
    assert "/blocks/{block_name}/run" in spec["paths"]


def test_openapi_spec_components():
    """Test OpenAPI spec components section."""
    spec = generate_openapi_spec()
    
    assert "schemas" in spec["components"]
    assert "FeatureResult" in spec["components"]["schemas"]
    assert "ErrorPolicy" in spec["components"]["schemas"]
    assert "ExecutionStatus" in spec["components"]["schemas"]


def test_openapi_spec_json_serializable():
    """Test that OpenAPI spec can be JSON serialized."""
    spec = generate_openapi_spec()
    
    # Should not raise
    json_str = json.dumps(spec)
    assert len(json_str) > 0
    
    # Should round-trip
    parsed = json.loads(json_str)
    assert parsed["openapi"] == spec["openapi"]


def test_introspect_feature():
    """Test feature introspection."""
    feature = SimpleFeature(name="test_feature")
    metadata = introspect_feature(feature)
    
    assert metadata["name"] == "test_feature"
    assert metadata["class"] == "SimpleFeature"
    assert "module" in metadata
    assert metadata["error_policy"] == "raise"


def test_introspect_feature_with_docstring():
    """Test feature introspection includes docstring."""
    feature = SimpleFeature(name="test")
    metadata = introspect_feature(feature)
    
    assert "description" in metadata
    assert "simple test feature" in metadata["description"].lower()


def test_introspect_feature_parameters():
    """Test feature introspection extracts parameters."""
    feature = SimpleFeature(name="test")
    metadata = introspect_feature(feature)
    
    assert "parameters" in metadata
    # Should detect window parameter from transform signature
    assert "window" in metadata["parameters"]


def test_introspect_block():
    """Test block introspection."""
    block = FeatureBlock([
        SimpleFeature(name="f1"),
        SimpleFeature(name="f2"),
    ], name="test_block")
    
    metadata = introspect_block(block)
    
    assert metadata["name"] == "test_block"
    assert metadata["class"] == "FeatureBlock"
    assert "features" in metadata
    assert len(metadata["features"]) == 2


def test_introspect_block_nested_features():
    """Test block introspection includes feature details."""
    block = FeatureBlock([
        SimpleFeature(name="feature1"),
    ], name="block1")
    
    metadata = introspect_block(block)
    feature_meta = metadata["features"][0]
    
    assert feature_meta["name"] == "feature1"
    assert feature_meta["class"] == "SimpleFeature"


def test_schema_validates_against_json_schema_draft():
    """Test that generated schema uses correct JSON Schema version."""
    schema = get_feature_result_schema()
    
    assert "$schema" in schema
    assert "json-schema.org" in schema["$schema"]


def test_feature_result_schema_metadata_type():
    """Test that metadata is typed as object."""
    schema = get_feature_result_schema()
    
    assert schema["properties"]["metadata"]["type"] == "object"
    assert schema["properties"]["metadata"]["additionalProperties"] is True


def test_feature_result_schema_trace_id_format():
    """Test that trace_id has uuid format."""
    schema = get_feature_result_schema()
    
    assert "format" in schema["properties"]["trace_id"]
    assert schema["properties"]["trace_id"]["format"] == "uuid"


def test_feature_result_schema_timestamp_format():
    """Test that timestamp has date-time format."""
    schema = get_feature_result_schema()
    
    assert "format" in schema["properties"]["timestamp"]
    assert schema["properties"]["timestamp"]["format"] == "date-time"
