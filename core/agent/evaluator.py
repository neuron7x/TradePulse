# SPDX-License-Identifier: MIT
"""High-throughput helpers for evaluating strategy populations."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .strategy import Strategy
from ..utils.metrics import get_metrics_collector

DatasetPreparer = Callable[[Any], Any]
ExecutorFactory = Callable[[int], Executor]


def _default_dataset_preparer(data: Any) -> Any:
    """Normalise raw market data once before strategy evaluation."""

    if data is None:
        return None

    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column for evaluation")
        return data.copy(deep=False)

    if isinstance(data, pd.Series):
        return pd.DataFrame({"close": data.astype(float).to_numpy(copy=False)})

    if isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError("Price array must be one-dimensional")
        return pd.DataFrame({"close": data.astype(float, copy=False)})

    if isinstance(data, (str, bytes)):
        raise TypeError("Unsupported dataset type for strategy evaluation")

    if isinstance(data, Iterable):
        return pd.DataFrame({"close": np.asarray(list(data), dtype=float)})

    raise TypeError("Unsupported dataset type for strategy evaluation")


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of a single strategy evaluation."""

    strategy: Strategy
    score: Optional[float]
    duration: float
    error: Optional[BaseException] = None

    @property
    def succeeded(self) -> bool:
        """Return ``True`` when the strategy finished without errors."""

        return self.error is None


class StrategyEvaluationError(RuntimeError):
    """Aggregate error raised when one or more strategies fail."""

    def __init__(self, failures: Sequence[EvaluationResult]):
        message = ", ".join(f"{res.strategy.name}: {res.error}" for res in failures)
        super().__init__(f"Strategy evaluation failed for {len(failures)} strategy(ies): {message}")
        self.failures = list(failures)


class StrategyBatchEvaluator:
    """Evaluate large strategy populations with bounded concurrency."""

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        chunk_size: int = 16,
        dataset_preparer: DatasetPreparer = _default_dataset_preparer,
        executor_factory: Optional[ExecutorFactory] = None,
        optimizer_label: str = "batch_evaluator",
    ) -> None:
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be positive when provided")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.dataset_preparer = dataset_preparer
        self.executor_factory = executor_factory or (
            lambda workers: ThreadPoolExecutor(max_workers=workers)
        )
        self.optimizer_label = optimizer_label

    def evaluate(
        self,
        strategies: Sequence[Strategy],
        data: Any,
        *,
        raise_on_error: bool = False,
    ) -> List[EvaluationResult]:
        """Evaluate ``strategies`` against ``data`` using a worker pool."""

        ordered_strategies = list(strategies)
        if not ordered_strategies:
            return []

        prepared_data = self.dataset_preparer(data)
        worker_count = self.max_workers or min(32, (os.cpu_count() or 1) + 4)
        collector = get_metrics_collector()

        results: List[Tuple[int, EvaluationResult]] = []
        failures: List[EvaluationResult] = []

        with self.executor_factory(worker_count) as executor:
            pending: dict[Future, Tuple[int, Strategy]] = {}
            iterator = enumerate(ordered_strategies)

            def _submit_next() -> bool:
                try:
                    idx, strategy = next(iterator)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._evaluate_strategy,
                    strategy,
                    prepared_data,
                    collector,
                )
                pending[future] = (idx, strategy)
                return True

            while len(pending) < self.chunk_size and _submit_next():
                pass

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, strategy = pending.pop(future)
                    result = future.result()
                    results.append((idx, result))
                    if result.error is not None:
                        failures.append(result)
                    while len(pending) < self.chunk_size and _submit_next():
                        pass

        results.sort(key=lambda item: item[0])
        ordered_results = [res for _, res in results]

        if raise_on_error and failures:
            raise StrategyEvaluationError(failures)

        return ordered_results

    def _evaluate_strategy(
        self,
        strategy: Strategy,
        data: Any,
        collector: Any,
    ) -> EvaluationResult:
        start = time.perf_counter()
        try:
            score = float(strategy.simulate_performance(data))
            duration = time.perf_counter() - start
            self._record_metrics(collector, strategy, score, duration, None)
            return EvaluationResult(strategy, score, duration, None)
        except BaseException as exc:  # pragma: no cover - defensive
            duration = time.perf_counter() - start
            self._record_metrics(collector, strategy, None, duration, exc)
            return EvaluationResult(strategy, None, duration, exc)

    def _record_metrics(
        self,
        collector: Any,
        strategy: Strategy,
        score: Optional[float],
        duration: float,
        error: Optional[BaseException],
    ) -> None:
        if not getattr(collector, "enabled", False):
            return

        try:
            collector.optimization_iterations.labels(
                optimizer_type=self.optimizer_label
            ).inc()
            collector.optimization_duration.labels(
                optimizer_type=self.optimizer_label
            ).observe(duration)
            if error is None and score is not None and math.isfinite(score):
                collector.set_strategy_score(strategy.name, score)
        except AttributeError:
            # Metrics collector initialised without Prometheus backend.
            pass


def evaluate_strategies(
    strategies: Sequence[Strategy],
    data: Any,
    *,
    max_workers: Optional[int] = None,
    chunk_size: int = 16,
    dataset_preparer: DatasetPreparer = _default_dataset_preparer,
) -> List[EvaluationResult]:
    """Convenience wrapper that evaluates strategies with default settings."""

    evaluator = StrategyBatchEvaluator(
        max_workers=max_workers,
        chunk_size=chunk_size,
        dataset_preparer=dataset_preparer,
    )
    return evaluator.evaluate(strategies, data)


__all__ = [
    "EvaluationResult",
    "StrategyBatchEvaluator",
    "StrategyEvaluationError",
    "evaluate_strategies",
]
