# SPDX-License-Identifier: MIT
"""Utilities for coordinating parallel strategy evaluations."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

from .evaluator import EvaluationResult, StrategyBatchEvaluator
from .strategy import Strategy


class _Evaluator(Protocol):
    def evaluate(
        self,
        strategies: Sequence[Strategy],
        data: Any,
        *,
        raise_on_error: bool = False,
    ) -> list[EvaluationResult]:
        """Execute the evaluation for *strategies* and return results."""


@dataclass(slots=True)
class StrategyFlow:
    """Container describing a batch of strategies to evaluate together."""

    name: str
    strategies: Sequence[Strategy]
    dataset: Any
    raise_on_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("StrategyFlow.name must be a non-empty string")
        if isinstance(self.strategies, (str, bytes)):
            raise TypeError("StrategyFlow.strategies must not be a string")
        if not isinstance(self.strategies, Sequence):
            raise TypeError("StrategyFlow.strategies must be a sequence of Strategy instances")

        strategies = tuple(self.strategies)
        if not strategies:
            raise ValueError("StrategyFlow must include at least one strategy")
        for strategy in strategies:
            if not isinstance(strategy, Strategy):
                raise TypeError("StrategyFlow.strategies must contain Strategy instances")
        object.__setattr__(self, "strategies", strategies)


class StrategyOrchestrationError(RuntimeError):
    """Aggregate failure raised when one or more flows fail."""

    def __init__(
        self,
        errors: Mapping[str, BaseException],
        results: Mapping[str, Sequence[EvaluationResult]],
    ) -> None:
        self.errors: Dict[str, BaseException] = dict(errors)
        self.results: Dict[str, Sequence[EvaluationResult]] = dict(results)
        message = ", ".join(f"{name}: {error}" for name, error in self.errors.items())
        super().__init__(
            f"Strategy orchestration failed for {len(self.errors)} flow(s): {message}"
        )


class StrategyOrchestrator:
    """Manage concurrent strategy evaluations with bounded parallelism."""

    def __init__(
        self,
        *,
        max_parallel: int | None = None,
        evaluator_factory: Callable[[], _Evaluator] | _Evaluator | None = None,
        thread_name_prefix: str = "strategy-orchestrator",
    ) -> None:
        if max_parallel is not None and max_parallel <= 0:
            raise ValueError("max_parallel must be positive when provided")

        workers = max_parallel or min(32, (os.cpu_count() or 1) + 4)
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._shutdown = False

        if evaluator_factory is None:
            self._factory: Callable[[], _Evaluator] = StrategyBatchEvaluator
        elif callable(evaluator_factory):
            self._factory = evaluator_factory  # type: ignore[assignment]
        elif hasattr(evaluator_factory, "evaluate"):
            self._factory = lambda: evaluator_factory  # type: ignore[assignment]
        else:  # pragma: no cover - defensive branch
            raise TypeError("evaluator_factory must be callable or expose an 'evaluate' method")

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def shutdown(self, *, wait: bool = True) -> None:
        """Terminate worker threads and reject new flows."""

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "StrategyOrchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Submission helpers
    def submit_flow(self, flow: StrategyFlow) -> Future[list[EvaluationResult]]:
        """Submit *flow* for asynchronous execution."""

        with self._lock:
            if self._shutdown:
                raise RuntimeError("StrategyOrchestrator has been shut down")
            if flow.name in self._active:
                raise RuntimeError(f"Flow '{flow.name}' is already running")
            self._active.add(flow.name)

        def _run() -> list[EvaluationResult]:
            try:
                evaluator = self._factory()
                return evaluator.evaluate(
                    flow.strategies,
                    flow.dataset,
                    raise_on_error=flow.raise_on_error,
                )
            finally:
                with self._lock:
                    self._active.discard(flow.name)

        return self._executor.submit(_run)

    def run_flows(
        self,
        flows: Sequence[StrategyFlow],
    ) -> Dict[str, list[EvaluationResult]]:
        """Execute *flows* concurrently and return collected results."""

        if not flows:
            return {}

        seen: set[str] = set()
        for flow in flows:
            if flow.name in seen:
                raise ValueError(f"Duplicate flow name detected: {flow.name}")
            seen.add(flow.name)

        futures: Dict[str, Future[list[EvaluationResult]]] = {
            flow.name: self.submit_flow(flow) for flow in flows
        }

        results: Dict[str, list[EvaluationResult]] = {}
        errors: Dict[str, BaseException] = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except BaseException as exc:  # pragma: no cover - defensive
                errors[name] = exc

        if errors:
            raise StrategyOrchestrationError(errors, results)
        return results

    # ------------------------------------------------------------------
    # Introspection helpers
    def active_flows(self) -> frozenset[str]:
        """Return a snapshot of currently executing flow names."""

        with self._lock:
            return frozenset(self._active)


__all__ = [
    "StrategyFlow",
    "StrategyOrchestrator",
    "StrategyOrchestrationError",
]
