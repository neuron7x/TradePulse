# SPDX-License-Identifier: MIT
"""JSON Schema generation for TradePulse data structures.

This module provides JSON schemas for all public payloads and integrates
with documentation generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Type, get_type_hints, get_origin, get_args


def dataclass_to_json_schema(cls: Type[Any], title: str | None = None) -> Dict[str, Any]:
    """Convert a dataclass to JSON Schema.
    
    Args:
        cls: Dataclass type to convert
        title: Optional schema title (defaults to class name)
        
    Returns:
        JSON Schema dictionary
        
    Example:
        >>> from core.indicators.base import FeatureResult
        >>> schema = dataclass_to_json_schema(FeatureResult)
        >>> print(json.dumps(schema, indent=2))
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
        
    schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": title or cls.__name__,
        "description": cls.__doc__ or f"Schema for {cls.__name__}",
        "properties": {},
        "required": [],
    }
    
    # Get type hints for the class
    hints = get_type_hints(cls)
    
    for field in fields(cls):
        field_name = field.name
        field_type = hints.get(field_name, field.type)
        
        # Build property schema
        prop_schema = _type_to_schema(field_type)
        
        # Add field metadata if available
        if field.metadata:
            prop_schema.update(field.metadata)
            
        schema["properties"][field_name] = prop_schema
        
        # Check if required (no default value)
        from dataclasses import MISSING
        if field.default is MISSING and field.default_factory is MISSING:
            schema["required"].append(field_name)
    
    # Clean up if no required fields
    if not schema["required"]:
        del schema["required"]
        
    return schema


def _type_to_schema(typ: Any) -> Dict[str, Any]:
    """Convert a Python type to JSON Schema type."""
    origin = get_origin(typ)
    args = get_args(typ)
    
    # Handle None/Optional
    if typ is type(None):
        return {"type": "null"}
        
    # Handle Union types (including Optional)
    if origin is type(None) or (origin and str(origin).startswith("typing.Union")):
        schemas = [_type_to_schema(arg) for arg in args if arg is not type(None)]
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}
        
    # Handle basic types
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }
    
    if typ in type_mapping:
        return type_mapping[typ]
        
    # Handle Dict types
    if origin is dict or typ is dict:
        result_schema: Dict[str, Any] = {"type": "object"}
        if args and len(args) == 2:
            # Dict[str, int] etc
            value_schema = _type_to_schema(args[1])
            result_schema["additionalProperties"] = value_schema
        return result_schema
        
    # Handle List/Sequence types
    if origin is list or origin is tuple or typ is list:
        list_schema: Dict[str, Any] = {"type": "array"}
        if args:
            items_schema = _type_to_schema(args[0])
            list_schema["items"] = items_schema
        return list_schema
        
    # Handle Mapping types
    if hasattr(typ, "__origin__") and "Mapping" in str(typ.__origin__):
        mapping_schema: Dict[str, Any] = {"type": "object"}
        if args and len(args) == 2:
            value_schema = _type_to_schema(args[1])
            mapping_schema["additionalProperties"] = value_schema
        return mapping_schema
        
    # Default to Any
    return {}


def generate_all_schemas() -> Dict[str, Dict[str, Any]]:
    """Generate schemas for all key TradePulse data structures.
    
    Returns:
        Dictionary mapping class names to schemas
    """
    from core.indicators.base import FeatureResult
    from backtest.engine import Result
    from core.data.ingestion import Ticker
    
    schemas = {}
    
    # Core schemas
    schemas["FeatureResult"] = dataclass_to_json_schema(
        FeatureResult,
        title="FeatureResult"
    )
    schemas["FeatureResult"]["description"] = (
        "Result from a feature/indicator transformation. Contains the computed "
        "value, metadata, and feature name."
    )
    
    schemas["BacktestResult"] = dataclass_to_json_schema(
        Result,
        title="BacktestResult"
    )
    schemas["BacktestResult"]["description"] = (
        "Result from a backtest run. Contains profit/loss, maximum drawdown, "
        "and number of trades."
    )
    
    schemas["Ticker"] = dataclass_to_json_schema(
        Ticker,
        title="Ticker"
    )
    schemas["Ticker"]["description"] = (
        "Market data tick with timestamp, price, and volume information."
    )
    
    return schemas


def save_schemas(output_dir: str = "docs/schemas") -> None:
    """Generate and save all schemas to JSON files.
    
    Args:
        output_dir: Directory to save schema files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    schemas = generate_all_schemas()
    
    for name, schema in schemas.items():
        filename = os.path.join(output_dir, f"{name}.json")
        with open(filename, "w") as f:
            json.dump(schema, f, indent=2)
            
    # Generate index file
    index = {
        "schemas": list(schemas.keys()),
        "version": "1.0.0",
        "base_url": "/schemas/",
    }
    
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate data against a JSON schema (basic validation).
    
    Args:
        data: Data dictionary to validate
        schema: JSON Schema to validate against
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check required fields
    if "required" in schema:
        for field in schema["required"]:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
    # Check property types
    if "properties" in schema:
        for key, value in data.items():
            if key not in schema["properties"]:
                continue
                
            prop_schema = schema["properties"][key]
            expected_type = prop_schema.get("type")
            
            if expected_type == "integer" and not isinstance(value, int):
                raise ValueError(f"Field {key} should be integer, got {type(value)}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"Field {key} should be number, got {type(value)}")
            elif expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Field {key} should be string, got {type(value)}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Field {key} should be boolean, got {type(value)}")
            elif expected_type == "array" and not isinstance(value, list):
                raise ValueError(f"Field {key} should be array, got {type(value)}")
            elif expected_type == "object" and not isinstance(value, dict):
                raise ValueError(f"Field {key} should be object, got {type(value)}")
                
    return True


__all__ = [
    "dataclass_to_json_schema",
    "generate_all_schemas",
    "save_schemas",
    "validate_against_schema",
]
