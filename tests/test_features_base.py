# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.base import (
    BaseFeature,
    FeatureBlock,
    FeatureResult,
    FunctionalFeature,
)


class _SquareFeature(BaseFeature):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name or "square")

    def transform(self, data, **_: object) -> FeatureResult:
        value = float(np.asarray(data).mean()) ** 2
        return FeatureResult(name=self.name, value=value, metadata={"op": "square"})


class _IdentityFeature(BaseFeature):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name)

    def transform(self, data, **_: object) -> FeatureResult:
        return FeatureResult(name=self.name, value=data, metadata={})


def test_feature_result_defaults_to_empty_metadata():
    result = FeatureResult(name="foo", value=1.0)
    assert result.metadata == {}


def test_base_feature_call_delegates_to_transform():
    feature = _SquareFeature()
    out = feature(np.array([1.0, 3.0]))
    assert pytest.approx(out.value, rel=1e-9) == 4.0
    assert out.metadata["op"] == "square"


def test_feature_block_register_and_extend_executes_all():
    block = FeatureBlock(
        [
            FunctionalFeature(sum, name="sum", metadata={"units": "raw"}),
        ]
    )
    block.register(FunctionalFeature(lambda data: max(data) - min(data), name="range"))
    block.extend([_SquareFeature(), _IdentityFeature(name="identity")])
    data = [1, 2, 3]
    result = block(data)
    assert set(result) == {"sum", "range", "square", "identity"}
    assert result["sum"] == 6
    assert result["identity"] == data


def test_functional_feature_preserves_metadata():
    feature = FunctionalFeature(
        lambda values: np.asarray(values).std(),
        name="std",
        metadata={"stat": "std"},
    )
    out = feature([1, 2, 3])
    assert out.name == "std"
    assert out.metadata == {"stat": "std"}


def test_feature_block_features_property_is_immutable():
    feature = _IdentityFeature(name="id")
    block = FeatureBlock([feature])
    features = block.features
    assert features == (feature,)
    with pytest.raises(AttributeError):
        features.append(feature)  # type: ignore[attr-defined]
