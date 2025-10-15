# SPDX-License-Identifier: MIT
"""Deterministic scheduler that periodically evaluates trading strategies."""
from __future__ import annotations

import os
import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

from .evaluator import EvaluationResult, StrategyBatchEvaluator
from .strategy import Strategy


LOGGER = logging.getLogger(__name__)


StrategyFactory = Callable[[], Sequence[Strategy]]
DatasetProvider = Callable[[], Any]
CompletionCallback = Callable[["StrategyJob", Sequence[EvaluationResult]], None]
ErrorCallback = Callable[["StrategyJob", BaseException], None]


@dataclass(slots=True)
class StrategyJob:
    """Configuration for a single scheduled strategy evaluation."""

    name: str
    strategies: Sequence[Strategy] | StrategyFactory
    data_provider: Any | DatasetProvider
    interval: float
    jitter: float = 0.0
    raise_on_error: bool = False
    enabled: bool = True
    on_complete: CompletionCallback | None = None
    on_error: ErrorCallback | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("StrategyJob.name must be a non-empty string")
        if self.interval <= 0:
            raise ValueError("StrategyJob.interval must be positive")
        if self.jitter < 0:
            raise ValueError("StrategyJob.jitter must be non-negative")
        if not callable(self.strategies):
            if not isinstance(self.strategies, Sequence):
                raise TypeError("StrategyJob.strategies must be a sequence or callable returning a sequence")
            if isinstance(self.strategies, (str, bytes)):
                raise TypeError("StrategyJob.strategies cannot be a string")
            strategies = list(self.strategies)
            if not strategies:
                raise ValueError("StrategyJob must include at least one strategy")
            for strategy in strategies:
                if not isinstance(strategy, Strategy):
                    raise TypeError("StrategyJob.strategies must contain Strategy instances")
            object.__setattr__(self, "strategies", tuple(strategies))
        if not callable(self.data_provider):
            # Allow static payloads (e.g., DataFrame, ndarray, or None).
            object.__setattr__(self, "data_provider", self.data_provider)

    def resolve_strategies(self) -> Sequence[Strategy]:
        """Return strategies to evaluate for this job."""

        strategies: Sequence[Strategy]
        if callable(self.strategies):
            strategies = list(self.strategies())
        else:
            strategies = list(self.strategies)
        if not strategies:
            raise ValueError(f"Strategy job '{self.name}' did not produce any strategies")
        for strategy in strategies:
            if not isinstance(strategy, Strategy):
                raise TypeError("Strategy factories must return Strategy instances")
        return strategies

    def resolve_dataset(self) -> Any:
        """Return the dataset to pass to the evaluator."""

        if callable(self.data_provider):
            return self.data_provider()
        return self.data_provider


@dataclass(slots=True)
class StrategyJobStatus:
    """Immutable snapshot of a scheduled job."""

    name: str
    enabled: bool
    next_run_at: float
    last_run_at: float | None
    consecutive_failures: int
    in_flight: bool
    last_error: BaseException | None
    result_count: int


@dataclass(slots=True)
class _JobState:
    job: StrategyJob
    next_run: float
    last_run: float | None = None
    last_results: tuple[EvaluationResult, ...] | None = None
    last_error: BaseException | None = None
    consecutive_failures: int = 0
    in_flight: bool = False
    future: Future[tuple[EvaluationResult, ...]] | None = None

    def schedule_next(self, *, base: float, rng: random.Random, interval: float | None = None) -> None:
        delay = interval if interval is not None else self.job.interval
        if self.job.jitter:
            jitter = rng.uniform(-self.job.jitter, self.job.jitter)
            delay = max(0.0, delay + jitter)
        self.next_run = base + max(0.0, delay)


class StrategyScheduler:
    """Coordinate periodic strategy evaluations."""

    def __init__(
        self,
        *,
        evaluator: StrategyBatchEvaluator | None = None,
        time_source: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
        max_backoff: float = 900.0,
        max_sleep: float = 5.0,
        idle_sleep: float = 0.5,
        max_workers: int | None = None,
    ) -> None:
        if max_backoff <= 0:
            raise ValueError("max_backoff must be positive")
        if max_sleep <= 0:
            raise ValueError("max_sleep must be positive")
        if idle_sleep <= 0:
            raise ValueError("idle_sleep must be positive")
        if max_workers is not None and max_workers <= 0:
            raise ValueError("max_workers must be positive when provided")

        self._evaluator = evaluator or StrategyBatchEvaluator()
        self._time = time_source or time.monotonic
        self._sleep = sleep or time.sleep
        self._rng = rng or random.Random()
        self._max_backoff = float(max_backoff)
        self._max_sleep = float(max_sleep)
        self._idle_sleep = float(idle_sleep)
        worker_count = max_workers or min(8, max(1, (os.cpu_count() or 1) * 2))
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="strategy-job")
        self._jobs: Dict[str, _JobState] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Registration & lifecycle management
    def add_job(self, job: StrategyJob, *, run_immediately: bool = False) -> None:
        """Register ``job`` with the scheduler."""

        with self._lock:
            if job.name in self._jobs:
                raise ValueError(f"Job '{job.name}' is already registered")
            now = self._time()
            state = _JobState(job=job, next_run=now)
            if run_immediately:
                state.next_run = now
            else:
                state.schedule_next(base=now, rng=self._rng)
            self._jobs[job.name] = state

    def remove_job(self, name: str) -> None:
        """Remove ``name`` from the scheduler."""

        with self._lock:
            if name not in self._jobs:
                raise KeyError(name)
            del self._jobs[name]

    def pause_job(self, name: str) -> None:
        """Temporarily disable a job without removing it."""

        with self._lock:
            state = self._jobs.get(name)
            if state is None:
                raise KeyError(name)
            state.job.enabled = False

    def resume_job(self, name: str, *, run_immediately: bool = False) -> None:
        """Re-enable a paused job."""

        with self._lock:
            state = self._jobs.get(name)
            if state is None:
                raise KeyError(name)
            state.job.enabled = True
            now = self._time()
            if run_immediately:
                state.next_run = now
            else:
                state.schedule_next(base=now, rng=self._rng)

    # ------------------------------------------------------------------
    # Execution
    def run_pending(self, *, wait: bool = True) -> Dict[str, list[EvaluationResult]]:
        """Execute all jobs whose schedule has elapsed.

        Args:
            wait: When ``True`` (default) block until each due job completes and
                return its evaluation results. When ``False`` the scheduler only
                dispatches work to the executor and returns immediately.

        Returns:
            Mapping of job name to a list of :class:`EvaluationResult` objects
            produced during this invocation. The mapping is empty when
            ``wait=False`` because work continues in the background.
        """

        now = self._time()
        with self._lock:
            due_states = [
                state
                for state in self._jobs.values()
                if state.job.enabled and not state.in_flight and state.next_run <= now
            ]
            for state in due_states:
                state.in_flight = True

        dispatch_order = sorted(due_states, key=lambda item: item.next_run)
        futures: list[tuple[_JobState, Future[tuple[EvaluationResult, ...]]]] = []
        for state in dispatch_order:
            future = self._submit_job(state)
            futures.append((state, future))

        if not wait:
            return {}

        results: Dict[str, list[EvaluationResult]] = {}
        for state, future in futures:
            try:
                outcome = future.result()
            except Exception:  # pragma: no cover - callback already recorded failure
                continue
            results[state.job.name] = list(outcome)
        return results

    def start(self, *, daemon: bool = True) -> None:
        """Start the background scheduling loop."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("StrategyScheduler already running")
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name="strategy-scheduler", daemon=daemon)
            self._thread.start()

    def stop(self, *, timeout: float | None = None) -> None:
        """Stop the background scheduling loop."""

        self._stop_event.set()
        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the scheduler and tear down worker threads."""

        self.stop()
        self._executor.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Introspection helpers
    def list_jobs(self) -> list[StrategyJobStatus]:
        """Return snapshots for all registered jobs."""

        with self._lock:
            return [self._snapshot(state) for state in self._jobs.values()]

    def get_status(self, name: str) -> StrategyJobStatus:
        """Return a snapshot for ``name``."""

        with self._lock:
            state = self._jobs.get(name)
            if state is None:
                raise KeyError(name)
            return self._snapshot(state)

    def get_last_results(self, name: str) -> tuple[EvaluationResult, ...] | None:
        """Return the most recent evaluation results for ``name``."""

        with self._lock:
            state = self._jobs.get(name)
            if state is None:
                raise KeyError(name)
            if state.last_results is None:
                return None
            return tuple(state.last_results)

    # ------------------------------------------------------------------
    # Internal helpers
    def _snapshot(self, state: _JobState) -> StrategyJobStatus:
        result_count = 0 if state.last_results is None else len(state.last_results)
        return StrategyJobStatus(
            name=state.job.name,
            enabled=state.job.enabled,
            next_run_at=state.next_run,
            last_run_at=state.last_run,
            consecutive_failures=state.consecutive_failures,
            in_flight=state.in_flight,
            last_error=state.last_error,
            result_count=result_count,
        )

    def _submit_job(self, state: _JobState) -> Future[tuple[EvaluationResult, ...]]:
        future = self._executor.submit(self._run_job, state)
        state.future = future
        future.add_done_callback(lambda fut, st=state: self._finalize_future(st, fut))
        return future

    def _run_job(self, state: _JobState) -> tuple[EvaluationResult, ...]:
        job = state.job
        strategies = job.resolve_strategies()
        dataset = job.resolve_dataset()
        evaluations = tuple(
            self._evaluator.evaluate(strategies, dataset, raise_on_error=job.raise_on_error)
        )
        return evaluations

    def _finalize_future(
        self, state: _JobState, future: Future[tuple[EvaluationResult, ...]]
    ) -> None:
        try:
            results = future.result()
        except BaseException as exc:  # pragma: no cover - defensive guard
            self._handle_failure(state, exc)
        else:
            self._handle_success(state, results)

    def _handle_success(self, state: _JobState, results: tuple[EvaluationResult, ...]) -> None:
        completed_at = self._time()
        job = state.job

        with self._lock:
            state.last_run = completed_at
            state.last_results = results
            state.last_error = None
            state.consecutive_failures = 0
            state.in_flight = False
            state.future = None
            state.schedule_next(base=completed_at, rng=self._rng)

        if job.on_complete is not None:
            try:
                job.on_complete(job, results)
            except Exception:  # pragma: no cover - defensive callback guard
                LOGGER.exception("Strategy job completion handler failed", extra={"job": job.name})

    def _handle_failure(self, state: _JobState, error: BaseException) -> None:
        failed_at = self._time()
        job = state.job

        with self._lock:
            state.last_run = failed_at
            state.last_results = None
            state.last_error = error
            state.consecutive_failures += 1
            state.in_flight = False
            state.future = None
            backoff = min(
                job.interval * (2 ** max(state.consecutive_failures - 1, 0)),
                self._max_backoff,
            )
            state.schedule_next(base=failed_at, rng=self._rng, interval=backoff)

        if job.on_error is not None:
            try:
                job.on_error(job, error)
            except Exception:  # pragma: no cover - defensive callback guard
                LOGGER.exception("Strategy job error handler failed", extra={"job": job.name})
        else:
            LOGGER.warning("Strategy job '%s' failed", job.name, exc_info=error)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_pending(wait=False)
            sleep_for = self._next_sleep_interval()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)

    def _next_sleep_interval(self) -> float:
        with self._lock:
            next_times = [
                state.next_run
                for state in self._jobs.values()
                if state.job.enabled and not state.in_flight
            ]
        if not next_times:
            return self._idle_sleep
        now = self._time()
        next_run = min(next_times)
        delay = max(0.0, next_run - now)
        return min(delay if delay > 0 else 0.0, self._max_sleep)


__all__ = ["StrategyJob", "StrategyJobStatus", "StrategyScheduler"]
