# SPDX-License-Identifier: MIT
from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ArrayLike",
    "FeatureInput",
    "FeatureOutput",
    "BaseFeature",
    "BaseBlock",
]

ArrayLike = np.ndarray | pd.Series | Sequence[float] | Iterable[float]
FeatureInput = Any
FeatureOutput = Any


class BaseFeature(ABC):
    """Abstract interface for reusable feature transforms.

    Concrete implementations should implement :meth:`transform` and may rely on
    :meth:`coerce_vector` to standardise 1-D numeric inputs. The interface is
    intentionally lightweight so that indicators implemented as simple
    functions can be promoted to composable, inspectable feature objects.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        params: Mapping[str, Any] | None = None,
        description: str | None = None,
    ) -> None:
        self._name = name or self.__class__.__name__
        self._params = dict(params or {})
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def params(self) -> Mapping[str, Any]:
        return MappingProxyType(self._params)

    def metadata(self) -> dict[str, Any]:
        """Return a serialisable description of the feature."""

        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "params": dict(self._params),
            "description": self._description,
        }

    def __call__(self, data: FeatureInput) -> FeatureOutput:
        return self.transform(data)

    @abstractmethod
    def transform(self, data: FeatureInput) -> FeatureOutput:
        """Compute the feature value for the provided *data*."""

    # NOTE: Sub-classes can rely on these helpers to normalise inputs.
    def coerce_vector(
        self,
        data: ArrayLike,
        *,
        dtype: type | np.dtype = float,
        allow_empty: bool = True,
    ) -> np.ndarray:
        """Return a 1-D numpy array from any recognised array-like input."""

        if isinstance(data, np.ndarray):
            vector = np.asarray(data, dtype=dtype)
        elif isinstance(data, pd.Series):
            vector = data.to_numpy(dtype=dtype, copy=False)
        else:
            vector = np.asarray(list(data), dtype=dtype)

        if vector.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects 1-D input, got shape {vector.shape}"
            )
        if not allow_empty and vector.size == 0:
            raise ValueError(f"{self.__class__.__name__} requires non-empty input")
        return vector


class BaseBlock(ABC):
    """Composable building block that orchestrates a set of features."""

    def __init__(
        self,
        *,
        name: str | None = None,
        features: Sequence[BaseFeature] | None = None,
        description: str | None = None,
    ) -> None:
        self._name = name or self.__class__.__name__
        self._features = tuple(features or ())
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        return self._features

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "description": self._description,
            "features": [feature.metadata() for feature in self._features],
        }

    def __call__(self, data: FeatureInput) -> Mapping[str, FeatureOutput]:
        return self.transform(data)

    @abstractmethod
    def transform(self, data: FeatureInput) -> Mapping[str, FeatureOutput]:
        """Execute the block against the supplied *data*."""
