# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from core.agent.evaluator import (
    EvaluationResult,
    StrategyBatchEvaluator,
    StrategyEvaluationError,
    evaluate_strategies,
)
from core.agent.strategy import Strategy


class _SleepyStrategy(Strategy):
    def __init__(self, name: str, delay: float) -> None:
        super().__init__(name=name, params={})
        self.delay = delay

    def simulate_performance(self, data: Any) -> float:  # pragma: no cover - exercised in tests
        time.sleep(self.delay)
        return self.delay


class _FailingStrategy(Strategy):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, params={})

    def simulate_performance(self, data: Any) -> float:  # pragma: no cover - exercised in tests
        raise RuntimeError(f"boom: {self.name}")


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"close": np.linspace(100.0, 101.0, 256)})


def test_batch_evaluator_preserves_strategy_order() -> None:
    strategies = [
        _SleepyStrategy("a", delay=0.0),
        _SleepyStrategy("b", delay=0.0),
        _SleepyStrategy("c", delay=0.0),
    ]

    evaluator = StrategyBatchEvaluator(max_workers=2)
    results = evaluator.evaluate(strategies, _sample_frame())

    assert [res.strategy.name for res in results] == ["a", "b", "c"]
    assert all(isinstance(res, EvaluationResult) for res in results)
    assert all(res.succeeded for res in results)
    assert all(res.score == pytest.approx(0.0, abs=1e-9) for res in results)


def test_batch_evaluator_parallelises_execution() -> None:
    strategies = [_SleepyStrategy(f"s{i}", delay=0.05) for i in range(4)]
    evaluator = StrategyBatchEvaluator(max_workers=4, chunk_size=4)

    start = time.perf_counter()
    results = evaluator.evaluate(strategies, _sample_frame())
    duration = time.perf_counter() - start

    assert duration < 0.15, f"Parallel evaluation was too slow ({duration:.3f}s)"
    assert all(res.score == pytest.approx(0.05, abs=1e-3) for res in results)


def test_batch_evaluator_reports_errors() -> None:
    strategies = [_SleepyStrategy("ok", delay=0.0), _FailingStrategy("bad")]
    evaluator = StrategyBatchEvaluator(max_workers=2)

    results = evaluator.evaluate(strategies, _sample_frame())
    assert len(results) == 2
    assert results[0].succeeded
    assert not results[1].succeeded
    assert isinstance(results[1].error, RuntimeError)

    with pytest.raises(StrategyEvaluationError) as excinfo:
        evaluator.evaluate(strategies, _sample_frame(), raise_on_error=True)
    assert len(excinfo.value.failures) == 1
    assert excinfo.value.failures[0].strategy.name == "bad"


def test_evaluate_strategies_helper_uses_preparer_once() -> None:
    counter = {"calls": 0}

    def _preparer(data: Any) -> Any:
        counter["calls"] += 1
        return data

    strategies = [_SleepyStrategy("a", delay=0.0), _SleepyStrategy("b", delay=0.0)]
    evaluate_strategies(strategies, _sample_frame(), dataset_preparer=_preparer)
    assert counter["calls"] == 1


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        StrategyBatchEvaluator(chunk_size=0)
