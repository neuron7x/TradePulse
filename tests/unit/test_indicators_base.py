# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.indicators.base import BaseFeature, FeatureBlock, FeatureResult, FunctionalFeature


class DoubleFeature(BaseFeature):
    def transform(self, data, **kwargs):
        return FeatureResult(name=self.name, value=float(data) * 2, metadata={})


@dataclass
class IncrementFeature(BaseFeature):
    increment: float = 1.0

    def __init__(self, increment: float = 1.0) -> None:
        super().__init__(name="increment")
        self.increment = increment

    def transform(self, data, **kwargs):
        return FeatureResult(name=self.name, value=float(data) + self.increment, metadata={})


def test_base_feature_callable_contract() -> None:
    feature = DoubleFeature(name="double")
    result = feature(3)
    assert result.value == 6.0
    assert result.name == "double"


def test_feature_block_executes_all_features() -> None:
    block = FeatureBlock([DoubleFeature(name="double")])
    block.register(IncrementFeature(increment=2.0))
    outputs = block.run(5)
    assert outputs["double"] == 10.0
    assert outputs["increment"] == 7.0


def test_functional_feature_wraps_callable() -> None:
    func_feature = FunctionalFeature(lambda x: np.sum(x), name="sum", metadata={"kind": "agg"})
    result = func_feature.transform(np.array([1, 2, 3]))
    assert result.value == 6
    assert result.metadata["kind"] == "agg"


def test_feature_block_extend() -> None:
    block = FeatureBlock()
    block.extend([DoubleFeature(name="double"), IncrementFeature(increment=0.5)])
    outputs = block(4)
    assert outputs == {"double": 8.0, "increment": 4.5}
