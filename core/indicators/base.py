# SPDX-License-Identifier: MIT
"""Foundational feature/block interfaces for indicator transformers.

These contracts make the fractal composition of indicators explicit: every
feature exposes the same `transform` signature and every block orchestrates a
homogeneous list of features.  Any new indicator can therefore be plugged into
an existing block (or nested block) without bespoke glue code.

This module provides:
- Strict type hints and Protocol definitions for all public APIs
- Runtime validation via pydantic for type safety
- Extensible Pre/Post processor interfaces
- Comprehensive error handling with provenance tracking
- Support for both synchronous and asynchronous operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableSequence,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)
from uuid import uuid4

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError:
    # Fallback if pydantic not available - use basic dataclasses
    BaseModel = None
    ConfigDict = None
    Field = None
    field_validator = None

import numpy as np

# Type aliases for clarity
FeatureInput = Union[np.ndarray, Any]
MetadataDict = Mapping[str, Any]

# Generic type variables
T = TypeVar("T")
ResultT = TypeVar("ResultT")


class ErrorPolicy(str, Enum):
    """Policy for handling errors during feature transformation."""

    RAISE = "raise"  # Raise exception immediately
    WARN = "warn"  # Log warning and return None
    SKIP = "skip"  # Skip silently and return None
    DEFAULT = "default"  # Return default value


class ExecutionStatus(str, Enum):
    """Status of feature transformation execution."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


# Pydantic models for strict validation (if available)
if BaseModel is not None:
    class FeatureResultModel(BaseModel):
        """Validated payload returned by feature transformers."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str = Field(..., description="Feature name/identifier")
        value: Any = Field(..., description="Computed feature value")
        metadata: dict[str, Any] = Field(
            default_factory=dict,
            description="Additional metadata about the computation"
        )
        status: ExecutionStatus = Field(
            default=ExecutionStatus.SUCCESS,
            description="Execution status"
        )
        error: Optional[str] = Field(
            default=None,
            description="Error message if status is FAILED"
        )
        trace_id: str = Field(
            default_factory=lambda: str(uuid4()),
            description="Unique trace identifier for debugging"
        )
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            description="Computation timestamp"
        )
        provenance: dict[str, Any] = Field(
            default_factory=dict,
            description="Audit trail: input hash, version, parameters"
        )

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Ensure name is not empty."""
            if not v or not v.strip():
                raise ValueError("Feature name cannot be empty")
            return v.strip()
else:
    # Fallback to dataclass without validation
    FeatureResultModel = None


@dataclass(slots=True)
class FeatureResult:
    """Canonical payload returned by every feature transformer.

    This is the lightweight version. For strict validation, use FeatureResultModel.
    """

    name: str
    value: Any
    metadata: MetadataDict = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_model(self) -> FeatureResultModel:
        """Convert to validated pydantic model if available."""
        if FeatureResultModel is None:
            raise RuntimeError("pydantic not available, cannot convert to model")
        return FeatureResultModel(
            name=self.name,
            value=self.value,
            metadata=dict(self.metadata),
            status=self.status,
            error=self.error,
            trace_id=self.trace_id,
            timestamp=self.timestamp,
            provenance=self.provenance,
        )

    def is_success(self) -> bool:
        """Check if transformation was successful."""
        return self.status == ExecutionStatus.SUCCESS

    def is_failed(self) -> bool:
        """Check if transformation failed."""
        return self.status == ExecutionStatus.FAILED


# Protocol definitions for extensibility
class PreProcessor(Protocol):
    """Protocol for preprocessing feature inputs before transformation."""

    def process(self, data: FeatureInput, **kwargs: Any) -> FeatureInput:
        """Preprocess input data before feature transformation.

        Args:
            data: Raw input data
            **kwargs: Additional parameters

        Returns:
            Preprocessed data ready for transformation
        """
        ...


class PostProcessor(Protocol):
    """Protocol for postprocessing feature results after transformation."""

    def process(self, result: FeatureResult, **kwargs: Any) -> FeatureResult:
        """Postprocess result after feature transformation.

        Args:
            result: Raw transformation result
            **kwargs: Additional parameters

        Returns:
            Postprocessed result
        """
        ...


class FeatureTransformer(Protocol):
    """Protocol defining the core feature transformation interface."""

    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Transform input data into a feature result.

        Args:
            data: Input data to transform
            **kwargs: Additional transformation parameters

        Returns:
            Feature result containing computed value and metadata
        """
        ...


class BaseFeature(ABC):
    """Structural contract for every indicator/feature transformer.

    All features must implement the transform method to convert input data
    into a FeatureResult. Features support preprocessing and postprocessing
    via optional processor hooks.

    Attributes:
        name: Feature identifier
        error_policy: How to handle transformation errors
        preprocessor: Optional input preprocessor
        postprocessor: Optional result postprocessor
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        error_policy: ErrorPolicy = ErrorPolicy.RAISE,
        preprocessor: Optional[PreProcessor] = None,
        postprocessor: Optional[PostProcessor] = None,
    ) -> None:
        """Initialize feature with name and optional processors.

        Args:
            name: Feature name (defaults to class name)
            error_policy: Policy for handling errors
            preprocessor: Optional input preprocessor
            postprocessor: Optional result postprocessor
        """
        self.name = name or self.__class__.__name__
        self.error_policy = error_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def __call__(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Make feature callable with same semantics as transform."""
        return self.transform(data, **kwargs)

    @abstractmethod
    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Produce a feature result from raw input.

        Args:
            data: Input data to transform
            **kwargs: Additional parameters

        Returns:
            FeatureResult with computed value and metadata
        """

    def _apply_preprocessor(
        self, data: FeatureInput, **kwargs: Any
    ) -> FeatureInput:
        """Apply preprocessor if configured."""
        if self.preprocessor is not None:
            return self.preprocessor.process(data, **kwargs)
        return data

    def _apply_postprocessor(
        self, result: FeatureResult, **kwargs: Any
    ) -> FeatureResult:
        """Apply postprocessor if configured."""
        if self.postprocessor is not None:
            return self.postprocessor.process(result, **kwargs)
        return result


class BaseBlock(ABC):
    """Composable container that orchestrates a homogeneous list of features.

    Blocks can contain features or other blocks, enabling fractal composition
    of indicator pipelines. All registered features are executed sequentially
    or in parallel depending on the block implementation.

    Attributes:
        name: Block identifier
        features: Immutable tuple of registered features
    """

    def __init__(
        self,
        features: Optional[Sequence[BaseFeature]] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        """Initialize block with optional feature list.

        Args:
            features: Initial features to register
            name: Block name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self._features: MutableSequence[BaseFeature] = list(features or [])

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        """Return immutable view of registered features."""
        return tuple(self._features)

    def register(self, feature: BaseFeature) -> None:
        """Register a single feature.

        Args:
            feature: Feature to register
        """
        self._features.append(feature)

    def extend(self, features: Iterable[BaseFeature]) -> None:
        """Register multiple features.

        Args:
            features: Features to register
        """
        self._features.extend(features)

    def __call__(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Make block callable with same semantics as run."""
        return self.run(data, **kwargs)

    @abstractmethod
    def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute the block over the input and return a feature mapping.

        Args:
            data: Input data for all features
            **kwargs: Additional parameters passed to features

        Returns:
            Mapping of feature names to computed values
        """


class FeatureBlock(BaseBlock):
    """Minimal block that executes its child features sequentially.

    This is the default block implementation that runs all registered
    features one after another and collects their results.
    """

    def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute all features sequentially.

        Args:
            data: Input data for all features
            **kwargs: Additional parameters passed to features

        Returns:
            Mapping of feature names to computed values
        """
        outputs: dict[str, Any] = {}
        for feature in self.features:
            result = feature.transform(data, **kwargs)
            outputs[result.name] = result.value
        return outputs


class FunctionalFeature(BaseFeature):
    """Adapter that wraps a plain function into the feature interface.

    This allows using simple functions as features without defining
    a full class. Useful for quick prototyping and functional composition.

    Example:
        >>> def mean_price(data): return np.mean(data)
        >>> feature = FunctionalFeature(mean_price, name="mean")
        >>> result = feature.transform(prices)
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        metadata: Optional[MetadataDict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize functional feature.

        Args:
            func: Function to wrap (signature: func(data, **kwargs) -> value)
            name: Feature name (defaults to function name)
            metadata: Static metadata to include in results
            **kwargs: Additional arguments passed to BaseFeature
        """
        super().__init__(name or func.__name__, **kwargs)
        self._func = func
        self._metadata = dict(metadata or {})

    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Transform by calling wrapped function.

        Args:
            data: Input data
            **kwargs: Additional parameters passed to function

        Returns:
            FeatureResult with function output as value
        """
        value = self._func(data, **kwargs)
        return FeatureResult(name=self.name, value=value, metadata=self._metadata)


__all__ = [
    "BaseFeature",
    "BaseBlock",
    "FeatureBlock",
    "FunctionalFeature",
    "FeatureResult",
    "FeatureResultModel",
    "ErrorPolicy",
    "ExecutionStatus",
    "PreProcessor",
    "PostProcessor",
    "FeatureTransformer",
    "FeatureInput",
    "MetadataDict",
]

