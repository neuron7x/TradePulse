# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.base import BaseFeature, FeatureResult, ParallelFeatureBlock
from core.indicators.entropy import entropy
from core.indicators.pipeline import IndicatorPipeline
from core.indicators.ricci import MeanRicciFeature, build_price_graph, mean_ricci


class _SumFeature(BaseFeature):
    def transform(self, data: np.ndarray, **_: object) -> FeatureResult:
        return FeatureResult(name=self.name, value=float(np.sum(data)), metadata={"buffer_id": id(data)})


def test_indicator_pipeline_reuses_float32_pool() -> None:
    series = np.arange(32, dtype=np.float64)
    features = [_SumFeature(name="sum_a"), _SumFeature(name="sum_b")]
    pipeline = IndicatorPipeline(features)

    first = pipeline.run(series)
    assert isinstance(first.values, dict)
    assert first.buffer.dtype == np.float32
    assert first.values["sum_a"] == first.values["sum_b"]
    buffer_id = id(first.buffer)
    first.release()

    second = pipeline.run(series)
    assert id(second.buffer) == buffer_id
    second.release()


def test_parallel_feature_block_thread_executes_all_features() -> None:
    block = ParallelFeatureBlock([_SumFeature(name="sum")], mode="thread")
    outputs = block.run(np.ones(8, dtype=np.float32))
    assert outputs["sum"] == pytest.approx(8.0)


def test_entropy_parallel_modes_match_cpu() -> None:
    rng = np.random.default_rng(42)
    series = rng.normal(size=2048)
    base_chunked = entropy(series, bins=64, chunk_size=256)
    process = entropy(series, bins=64, chunk_size=256, parallel="process")
    async_value = entropy(series, bins=64, chunk_size=256, parallel="async")
    gpu_fallback = entropy(series, bins=64, backend="gpu")

    assert process == pytest.approx(base_chunked, rel=1e-6)
    assert async_value == pytest.approx(base_chunked, rel=1e-6)
    assert gpu_fallback == pytest.approx(entropy(series, bins=64), rel=1e-6)


def test_mean_ricci_async_matches_sequential() -> None:
    prices = np.linspace(100, 110, num=40)
    graph = build_price_graph(prices, delta=0.01)
    sequential = mean_ricci(graph)
    async_result = mean_ricci(graph, parallel="async")
    assert async_result == pytest.approx(sequential, rel=1e-6)


def test_mean_ricci_feature_async_metadata() -> None:
    feature = MeanRicciFeature(delta=0.01, parallel_async=True, max_workers=2)
    prices = np.linspace(50, 55, num=30)
    result = feature.transform(prices)
    assert result.metadata["parallel"] == "async"
    assert result.metadata["max_workers"] == 2
