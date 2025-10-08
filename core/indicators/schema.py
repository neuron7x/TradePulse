# SPDX-License-Identifier: MIT
"""Schema generation for indicators API.

This module provides JSON Schema and OpenAPI schema generation for
indicator interfaces, enabling automatic API documentation and
validation for integrations.

Features:
- JSON Schema generation for FeatureResult and related types
- OpenAPI 3.0 schema generation for REST/RPC interfaces
- Type introspection and automatic schema inference
- Pydantic integration when available
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from .base import (
    BaseBlock,
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureResultModel,
)


def get_feature_result_schema() -> Dict[str, Any]:
    """Generate JSON Schema for FeatureResult.

    Returns a JSON Schema (draft 2020-12) document describing the
    FeatureResult structure, suitable for API documentation and
    validation.

    Returns:
        JSON Schema dictionary

    Example:
        >>> schema = get_feature_result_schema()
        >>> print(schema["properties"]["name"]["type"])
        string
    """
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FeatureResult",
        "description": "Result payload from feature transformation",
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Feature name/identifier",
                "minLength": 1,
            },
            "value": {
                "description": "Computed feature value (any type)",
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata about the computation",
                "additionalProperties": True,
            },
            "status": {
                "type": "string",
                "description": "Execution status",
                "enum": [status.value for status in ExecutionStatus],
            },
            "error": {
                "type": ["string", "null"],
                "description": "Error message if status is FAILED",
            },
            "trace_id": {
                "type": "string",
                "description": "Unique trace identifier for debugging",
                "format": "uuid",
            },
            "timestamp": {
                "type": "string",
                "description": "Computation timestamp",
                "format": "date-time",
            },
            "provenance": {
                "type": "object",
                "description": "Audit trail: input hash, version, parameters",
                "additionalProperties": True,
            },
        },
        "required": ["name", "value", "metadata", "status", "trace_id", "timestamp"],
    }

    return schema


def get_pydantic_schema() -> Optional[Dict[str, Any]]:
    """Generate JSON Schema using pydantic model if available.

    Returns pydantic-generated schema which includes additional
    validation rules and constraints.

    Returns:
        JSON Schema dictionary or None if pydantic unavailable
    """
    if FeatureResultModel is None:
        return None

    try:
        # Pydantic v2 API
        if hasattr(FeatureResultModel, "model_json_schema"):
            return FeatureResultModel.model_json_schema()
        # Pydantic v1 API
        elif hasattr(FeatureResultModel, "schema"):
            return FeatureResultModel.schema()
    except Exception:
        pass

    return None


def get_error_policy_schema() -> Dict[str, Any]:
    """Generate JSON Schema for ErrorPolicy enum.

    Returns:
        JSON Schema dictionary
    """
    return {
        "title": "ErrorPolicy",
        "description": "Policy for handling errors during transformation",
        "type": "string",
        "enum": [policy.value for policy in ErrorPolicy],
    }


def get_execution_status_schema() -> Dict[str, Any]:
    """Generate JSON Schema for ExecutionStatus enum.

    Returns:
        JSON Schema dictionary
    """
    return {
        "title": "ExecutionStatus",
        "description": "Status of feature transformation execution",
        "type": "string",
        "enum": [status.value for status in ExecutionStatus],
    }


def get_feature_transform_operation_schema() -> Dict[str, Any]:
    """Generate OpenAPI operation schema for feature transform.

    Returns OpenAPI 3.0 operation object describing the transform
    endpoint, including request body and response schemas.

    Returns:
        OpenAPI operation dictionary
    """
    return {
        "summary": "Transform input data using feature",
        "description": (
            "Apply feature transformation to input data and return result. "
            "The transformation is idempotent for deterministic features."
        ),
        "operationId": "transformFeature",
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "description": "Input data (structure varies by feature)",
                            },
                            "kwargs": {
                                "type": "object",
                                "description": "Additional transformation parameters",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["data"],
                    },
                }
            },
        },
        "responses": {
            "200": {
                "description": "Successful transformation",
                "content": {
                    "application/json": {
                        "schema": get_feature_result_schema(),
                    }
                },
            },
            "400": {
                "description": "Invalid input data",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "500": {
                "description": "Transformation error",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "trace_id": {"type": "string"},
                            },
                        }
                    }
                },
            },
        },
    }


def get_block_run_operation_schema() -> Dict[str, Any]:
    """Generate OpenAPI operation schema for block run.

    Returns OpenAPI 3.0 operation object describing the block run
    endpoint, including request body and response schemas.

    Returns:
        OpenAPI operation dictionary
    """
    return {
        "summary": "Execute feature block on input data",
        "description": (
            "Run all features in the block sequentially or concurrently "
            "and return aggregated results."
        ),
        "operationId": "runFeatureBlock",
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "description": "Input data for all features",
                            },
                            "kwargs": {
                                "type": "object",
                                "description": "Parameters passed to all features",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["data"],
                    },
                }
            },
        },
        "responses": {
            "200": {
                "description": "Successful block execution",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "description": "Mapping of feature names to values",
                            "additionalProperties": True,
                        },
                    }
                },
            },
            "400": {
                "description": "Invalid input data",
            },
            "500": {
                "description": "Block execution error",
            },
        },
    }


def generate_openapi_spec(
    title: str = "TradePulse Indicators API",
    version: str = "1.0.0",
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate complete OpenAPI 3.0 specification for indicators API.

    Args:
        title: API title
        version: API version
        description: Optional API description

    Returns:
        Complete OpenAPI 3.0 specification dictionary

    Example:
        >>> spec = generate_openapi_spec()
        >>> print(spec["openapi"])
        3.0.3
    """
    if description is None:
        description = (
            "REST API for TradePulse indicator transformations. "
            "Provides endpoints for executing features and blocks "
            "with comprehensive type validation and error handling."
        )

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": description,
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
        },
        "servers": [
            {
                "url": "http://localhost:8000/api/v1",
                "description": "Local development server",
            },
        ],
        "paths": {
            "/features/{feature_name}/transform": {
                "post": get_feature_transform_operation_schema(),
            },
            "/blocks/{block_name}/run": {
                "post": get_block_run_operation_schema(),
            },
        },
        "components": {
            "schemas": {
                "FeatureResult": get_feature_result_schema(),
                "ErrorPolicy": get_error_policy_schema(),
                "ExecutionStatus": get_execution_status_schema(),
            },
        },
    }

    return spec


def introspect_feature(feature: BaseFeature) -> Dict[str, Any]:
    """Introspect a feature class and generate metadata.

    Extracts metadata from feature class including parameters,
    docstrings, and type hints for automatic documentation.

    Args:
        feature: Feature instance to introspect

    Returns:
        Feature metadata dictionary
    """
    metadata = {
        "name": feature.name,
        "class": feature.__class__.__name__,
        "module": feature.__class__.__module__,
        "error_policy": feature.error_policy.value,
    }

    # Extract docstring
    if feature.__class__.__doc__:
        metadata["description"] = inspect.cleandoc(feature.__class__.__doc__)

    # Extract transform method signature
    try:
        sig = inspect.signature(feature.transform)
        metadata["parameters"] = {}
        for param_name, param in sig.parameters.items():
            if param_name not in ("self", "data"):
                param_info = {"name": param_name}
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = str(param.default)
                metadata["parameters"][param_name] = param_info
    except Exception:
        pass

    return metadata


def introspect_block(block: BaseBlock) -> Dict[str, Any]:
    """Introspect a block and generate metadata.

    Args:
        block: Block instance to introspect

    Returns:
        Block metadata dictionary
    """
    metadata = {
        "name": block.name,
        "class": block.__class__.__name__,
        "module": block.__class__.__module__,
        "features": [
            introspect_feature(feature)
            for feature in block.features
        ],
    }

    if block.__class__.__doc__:
        metadata["description"] = inspect.cleandoc(block.__class__.__doc__)

    return metadata


__all__ = [
    "get_feature_result_schema",
    "get_pydantic_schema",
    "get_error_policy_schema",
    "get_execution_status_schema",
    "get_feature_transform_operation_schema",
    "get_block_run_operation_schema",
    "generate_openapi_spec",
    "introspect_feature",
    "introspect_block",
]
