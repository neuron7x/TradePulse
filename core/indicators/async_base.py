# SPDX-License-Identifier: MIT
"""Async support for feature transformers.

This module provides async versions of the core feature/block interfaces,
enabling non-blocking I/O operations and concurrent execution of indicators.

All async features maintain the same contract as their sync counterparts,
making it easy to swap between sync and async implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Protocol, Sequence

from .base import (
    BaseFeature,
    ErrorPolicy,
    FeatureInput,
    FeatureResult,
)


class AsyncPreProcessor(Protocol):
    """Protocol for async preprocessing of feature inputs."""

    async def process(self, data: FeatureInput, **kwargs: Any) -> FeatureInput:
        """Preprocess input data asynchronously.

        Args:
            data: Raw input data
            **kwargs: Additional parameters

        Returns:
            Preprocessed data ready for transformation
        """
        ...


class AsyncPostProcessor(Protocol):
    """Protocol for async postprocessing of feature results."""

    async def process(self, result: FeatureResult, **kwargs: Any) -> FeatureResult:
        """Postprocess result asynchronously.

        Args:
            result: Raw transformation result
            **kwargs: Additional parameters

        Returns:
            Postprocessed result
        """
        ...


class AsyncFeatureTransformer(Protocol):
    """Protocol defining async feature transformation interface."""

    async def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Transform input data asynchronously.

        Args:
            data: Input data to transform
            **kwargs: Additional transformation parameters

        Returns:
            Feature result containing computed value and metadata
        """
        ...


class BaseFeatureAsync(ABC):
    """Async version of BaseFeature.

    All async features must implement the async transform method.
    Features can optionally have async preprocessors and postprocessors.

    Attributes:
        name: Feature identifier
        error_policy: How to handle transformation errors
        preprocessor: Optional async input preprocessor
        postprocessor: Optional async result postprocessor
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        error_policy: ErrorPolicy = ErrorPolicy.RAISE,
        preprocessor: Optional[AsyncPreProcessor] = None,
        postprocessor: Optional[AsyncPostProcessor] = None,
    ) -> None:
        """Initialize async feature.

        Args:
            name: Feature name (defaults to class name)
            error_policy: Policy for handling errors
            preprocessor: Optional async input preprocessor
            postprocessor: Optional async result postprocessor
        """
        self.name = name or self.__class__.__name__
        self.error_policy = error_policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    async def __call__(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Make feature async callable."""
        return await self.transform(data, **kwargs)

    @abstractmethod
    async def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Produce a feature result asynchronously.

        Args:
            data: Input data to transform
            **kwargs: Additional parameters

        Returns:
            FeatureResult with computed value and metadata
        """

    async def _apply_preprocessor(
        self, data: FeatureInput, **kwargs: Any
    ) -> FeatureInput:
        """Apply async preprocessor if configured."""
        if self.preprocessor is not None:
            return await self.preprocessor.process(data, **kwargs)
        return data

    async def _apply_postprocessor(
        self, result: FeatureResult, **kwargs: Any
    ) -> FeatureResult:
        """Apply async postprocessor if configured."""
        if self.postprocessor is not None:
            return await self.postprocessor.process(result, **kwargs)
        return result


class BaseBlockAsync(ABC):
    """Async version of BaseBlock.

    Async blocks orchestrate async features and can execute them
    sequentially or concurrently depending on the implementation.

    Attributes:
        name: Block identifier
        features: Immutable tuple of registered async features
    """

    def __init__(
        self,
        features: Optional[Sequence[BaseFeatureAsync]] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        """Initialize async block.

        Args:
            features: Initial async features to register
            name: Block name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self._features: MutableSequence[BaseFeatureAsync] = list(features or [])

    @property
    def features(self) -> tuple[BaseFeatureAsync, ...]:
        """Return immutable view of registered features."""
        return tuple(self._features)

    def register(self, feature: BaseFeatureAsync) -> None:
        """Register a single async feature.

        Args:
            feature: Async feature to register
        """
        self._features.append(feature)

    def extend(self, features: Iterable[BaseFeatureAsync]) -> None:
        """Register multiple async features.

        Args:
            features: Async features to register
        """
        self._features.extend(features)

    async def __call__(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Make block async callable."""
        return await self.run(data, **kwargs)

    @abstractmethod
    async def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute the block asynchronously.

        Args:
            data: Input data for all features
            **kwargs: Additional parameters passed to features

        Returns:
            Mapping of feature names to computed values
        """


class FeatureBlockAsync(BaseBlockAsync):
    """Async block that executes features sequentially.

    This is the default async block that runs features one after
    another using async/await.
    """

    async def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute all features sequentially with async/await.

        Args:
            data: Input data for all features
            **kwargs: Additional parameters passed to features

        Returns:
            Mapping of feature names to computed values
        """
        outputs: dict[str, Any] = {}
        for feature in self.features:
            result = await feature.transform(data, **kwargs)
            outputs[result.name] = result.value
        return outputs


class FeatureBlockConcurrent(BaseBlockAsync):
    """Async block that executes features concurrently.

    Uses asyncio.gather to run all features in parallel,
    significantly improving performance when features are I/O bound.
    """

    async def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute all features concurrently.

        Args:
            data: Input data for all features
            **kwargs: Additional parameters passed to features

        Returns:
            Mapping of feature names to computed values
        """
        import asyncio

        # Execute all features concurrently
        results = await asyncio.gather(
            *[feature.transform(data, **kwargs) for feature in self.features]
        )

        # Collect results
        outputs: dict[str, Any] = {}
        for result in results:
            outputs[result.name] = result.value
        return outputs


class AsyncFeatureAdapter(BaseFeatureAsync):
    """Adapter to run sync features in async context.

    Wraps a synchronous feature and executes it in an async executor,
    allowing sync features to be used in async pipelines.
    """

    def __init__(
        self,
        sync_feature: BaseFeature,
        **kwargs: Any,
    ) -> None:
        """Initialize async adapter for sync feature.

        Args:
            sync_feature: Synchronous feature to wrap
            **kwargs: Additional arguments passed to BaseFeatureAsync
        """
        super().__init__(name=sync_feature.name, **kwargs)
        self._sync_feature = sync_feature

    async def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Execute sync feature in async executor.

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            FeatureResult from sync feature
        """
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sync_feature.transform(data, **kwargs)
        )
        return result


__all__ = [
    "AsyncPreProcessor",
    "AsyncPostProcessor",
    "AsyncFeatureTransformer",
    "BaseFeatureAsync",
    "BaseBlockAsync",
    "FeatureBlockAsync",
    "FeatureBlockConcurrent",
    "AsyncFeatureAdapter",
]
