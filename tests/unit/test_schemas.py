# SPDX-License-Identifier: MIT
"""Tests for JSON Schema generation module."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest

from core.utils.schemas import (
    dataclass_to_json_schema,
    generate_all_schemas,
    save_schemas,
    validate_against_schema,
)


@dataclass
class SimpleTestClass:
    """Simple test dataclass."""
    name: str
    age: int
    score: float


@dataclass
class OptionalFieldsClass:
    """Test dataclass with optional fields."""
    required_field: str
    optional_field: Optional[int] = None
    default_value: str = "default"


@dataclass
class ComplexTypesClass:
    """Test dataclass with complex types."""
    tags: List[str]
    metadata: Dict[str, Any]
    scores: List[float] = field(default_factory=list)


class TestDataclassToJsonSchema:
    """Test dataclass to JSON schema conversion."""

    def test_simple_dataclass_schema(self) -> None:
        """Should generate schema for simple dataclass."""
        schema = dataclass_to_json_schema(SimpleTestClass)
        
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["title"] == "SimpleTestClass"
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "score" in schema["properties"]
        
        # Check types
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"
        assert schema["properties"]["score"]["type"] == "number"
        
        # All fields should be required (no defaults)
        assert set(schema["required"]) == {"name", "age", "score"}

    def test_optional_fields_not_required(self) -> None:
        """Optional and default fields should not be required."""
        schema = dataclass_to_json_schema(OptionalFieldsClass)
        
        # Only required_field has no default
        assert schema["required"] == ["required_field"]
        assert "optional_field" in schema["properties"]
        assert "default_value" in schema["properties"]

    def test_custom_title(self) -> None:
        """Should use custom title when provided."""
        schema = dataclass_to_json_schema(SimpleTestClass, title="CustomTitle")
        assert schema["title"] == "CustomTitle"

    def test_complex_types_schema(self) -> None:
        """Should handle complex types like List and Dict."""
        schema = dataclass_to_json_schema(ComplexTypesClass)
        
        # Check List[str]
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["tags"]["items"]["type"] == "string"
        
        # Check Dict[str, Any]
        assert schema["properties"]["metadata"]["type"] == "object"
        
        # Check List[float] with default_factory
        assert schema["properties"]["scores"]["type"] == "array"
        assert schema["properties"]["scores"]["items"]["type"] == "number"

    def test_non_dataclass_raises_error(self) -> None:
        """Should raise TypeError for non-dataclass."""
        class NotADataclass:
            pass
        
        with pytest.raises(TypeError, match="is not a dataclass"):
            dataclass_to_json_schema(NotADataclass)

    def test_no_required_fields_omitted(self) -> None:
        """If all fields have defaults, required should be omitted."""
        @dataclass
        class AllDefaults:
            field1: str = "default"
            field2: int = 42
        
        schema = dataclass_to_json_schema(AllDefaults)
        assert "required" not in schema


class TestGenerateAllSchemas:
    """Test generation of all TradePulse schemas."""

    def test_generates_all_core_schemas(self) -> None:
        """Should generate schemas for all core types."""
        schemas = generate_all_schemas()
        
        assert "FeatureResult" in schemas
        assert "BacktestResult" in schemas
        assert "Ticker" in schemas
        
        # Each should be a valid schema
        for name, schema in schemas.items():
            assert "$schema" in schema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_feature_result_schema_complete(self) -> None:
        """FeatureResult schema should have all expected properties."""
        schemas = generate_all_schemas()
        schema = schemas["FeatureResult"]
        
        assert "value" in schema["properties"]
        assert "metadata" in schema["properties"]
        assert "name" in schema["properties"]

    def test_backtest_result_schema_complete(self) -> None:
        """BacktestResult schema should have all expected properties."""
        schemas = generate_all_schemas()
        schema = schemas["BacktestResult"]
        
        # Should have profit/loss related fields
        assert "properties" in schema
        # Result dataclass has: pnl, max_dd, trades
        assert "pnl" in schema["properties"]
        assert "max_dd" in schema["properties"]
        assert "trades" in schema["properties"]

    def test_ticker_schema_complete(self) -> None:
        """Ticker schema should have all expected properties."""
        schemas = generate_all_schemas()
        schema = schemas["Ticker"]
        
        assert "ts" in schema["properties"]
        assert "price" in schema["properties"]
        assert "volume" in schema["properties"]


class TestSaveSchemas:
    """Test saving schemas to files."""

    def test_saves_schemas_to_directory(self) -> None:
        """Should save all schemas as JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_schemas(tmpdir)
            
            # Check files exist
            assert Path(tmpdir, "FeatureResult.json").exists()
            assert Path(tmpdir, "BacktestResult.json").exists()
            assert Path(tmpdir, "Ticker.json").exists()
            assert Path(tmpdir, "index.json").exists()

    def test_saved_schemas_valid_json(self) -> None:
        """Saved schemas should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_schemas(tmpdir)
            
            # Load and verify each schema
            for filename in ["FeatureResult.json", "BacktestResult.json", "Ticker.json"]:
                filepath = Path(tmpdir, filename)
                with open(filepath) as f:
                    schema = json.load(f)
                assert "$schema" in schema
                assert "type" in schema

    def test_creates_index_file(self) -> None:
        """Should create an index file listing all schemas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_schemas(tmpdir)
            
            index_path = Path(tmpdir, "index.json")
            with open(index_path) as f:
                index = json.load(f)
            
            assert "schemas" in index
            assert "version" in index
            assert "base_url" in index
            assert len(index["schemas"]) >= 3

    def test_creates_output_directory(self) -> None:
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir, "nested", "schemas")
            assert not output_dir.exists()
            
            save_schemas(str(output_dir))
            
            assert output_dir.exists()
            assert Path(output_dir, "index.json").exists()


class TestValidateAgainstSchema:
    """Test schema validation."""

    def test_valid_data_passes(self) -> None:
        """Valid data should pass validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"]
        }
        
        data = {"name": "Alice", "age": 30}
        assert validate_against_schema(data, schema) is True

    def test_missing_required_field_fails(self) -> None:
        """Missing required field should raise ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"]
        }
        
        data = {"name": "Alice"}  # Missing age
        with pytest.raises(ValueError, match="Missing required field: age"):
            validate_against_schema(data, schema)

    def test_wrong_type_fails(self) -> None:
        """Wrong type should raise ValueError."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        
        data = {"age": "not an integer"}
        with pytest.raises(ValueError, match="should be integer"):
            validate_against_schema(data, schema)

    def test_validates_string_type(self) -> None:
        """Should validate string types."""
        schema = {
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        with pytest.raises(ValueError, match="should be string"):
            validate_against_schema({"name": 123}, schema)

    def test_validates_number_type(self) -> None:
        """Should validate number types (int or float)."""
        schema = {
            "properties": {
                "score": {"type": "number"}
            }
        }
        
        # Both int and float should be valid for number
        assert validate_against_schema({"score": 42}, schema) is True
        assert validate_against_schema({"score": 42.5}, schema) is True
        
        # String should fail
        with pytest.raises(ValueError, match="should be number"):
            validate_against_schema({"score": "not a number"}, schema)

    def test_validates_boolean_type(self) -> None:
        """Should validate boolean types."""
        schema = {
            "properties": {
                "active": {"type": "boolean"}
            }
        }
        
        assert validate_against_schema({"active": True}, schema) is True
        
        with pytest.raises(ValueError, match="should be boolean"):
            validate_against_schema({"active": "true"}, schema)

    def test_validates_array_type(self) -> None:
        """Should validate array types."""
        schema = {
            "properties": {
                "items": {"type": "array"}
            }
        }
        
        assert validate_against_schema({"items": [1, 2, 3]}, schema) is True
        
        with pytest.raises(ValueError, match="should be array"):
            validate_against_schema({"items": "not an array"}, schema)

    def test_validates_object_type(self) -> None:
        """Should validate object types."""
        schema = {
            "properties": {
                "config": {"type": "object"}
            }
        }
        
        assert validate_against_schema({"config": {"key": "value"}}, schema) is True
        
        with pytest.raises(ValueError, match="should be object"):
            validate_against_schema({"config": "not an object"}, schema)

    def test_extra_fields_allowed(self) -> None:
        """Extra fields not in schema should be allowed."""
        schema = {
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        data = {"name": "Alice", "extra": "field"}
        assert validate_against_schema(data, schema) is True

    def test_no_properties_schema(self) -> None:
        """Schema without properties should not fail."""
        schema = {"type": "object"}
        data = {"anything": "goes"}
        assert validate_against_schema(data, schema) is True

    def test_no_required_schema(self) -> None:
        """Schema without required fields should not fail."""
        schema = {
            "properties": {
                "optional": {"type": "string"}
            }
        }
        data = {}
        assert validate_against_schema(data, schema) is True
