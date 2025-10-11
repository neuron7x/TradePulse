# SPDX-License-Identifier: MIT
"""Foundational feature/block interfaces for indicator transformers.

These contracts make the fractal composition of indicators explicit: every
feature exposes the same `transform` signature and every block orchestrates a
homogeneous list of features.  Any new indicator can therefore be plugged into
an existing block (or nested block) without bespoke glue code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableSequence, Sequence

from core.utils.metrics import get_metrics_collector
from observability.tracing import pipeline_span

FeatureInput = Any


@dataclass(slots=True)
class FeatureResult:
    """Canonical payload returned by every feature transformer."""

    name: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseFeature(ABC):
    """Structural contract for every indicator/feature transformer."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def __call__(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        return self.transform(data, **kwargs)

    @abstractmethod
    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Produce a feature result from raw input."""
        
    def transform_with_metrics(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        """Transform with automatic metrics collection."""
        metrics = get_metrics_collector()
        feature_type = kwargs.get("feature_type", "generic")
        
        with pipeline_span("features.transform", feature_name=self.name, feature_type=feature_type):
            with metrics.measure_feature_transform(self.name, feature_type):
                result = self.transform(data, **kwargs)
            
        # Record the feature value if it's numeric
        if isinstance(result.value, (int, float)):
            metrics.record_feature_value(self.name, float(result.value))
            
        return result


class BaseBlock(ABC):
    """Composable container that orchestrates a homogeneous list of features."""

    def __init__(
        self,
        features: Sequence[BaseFeature] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self._features: MutableSequence[BaseFeature] = list(features or [])

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        return tuple(self._features)

    def register(self, feature: BaseFeature) -> None:
        self._features.append(feature)

    def extend(self, features: Iterable[BaseFeature]) -> None:
        self._features.extend(features)

    def __call__(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        return self.run(data, **kwargs)

    @abstractmethod
    def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        """Execute the block over the input and return a feature mapping."""


class FeatureBlock(BaseBlock):
    """Minimal block that executes its child features sequentially."""

    def run(self, data: FeatureInput, **kwargs: Any) -> Mapping[str, Any]:
        outputs: dict[str, Any] = {}
        for feature in self.features:
            result = feature.transform(data, **kwargs)
            outputs[result.name] = result.value
        return outputs


class FunctionalFeature(BaseFeature):
    """Adapter that wraps a plain function into the feature interface."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self._func = func
        self._metadata = dict(metadata or {})

    def transform(self, data: FeatureInput, **kwargs: Any) -> FeatureResult:
        value = self._func(data, **kwargs)
        return FeatureResult(name=self.name, value=value, metadata=self._metadata)


__all__ = [
    "BaseFeature",
    "BaseBlock",
    "FeatureBlock",
    "FunctionalFeature",
    "FeatureResult",
]

