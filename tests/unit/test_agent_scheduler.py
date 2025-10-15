"""Tests for the strategy scheduler."""
from __future__ import annotations

import random
import threading
import time
from typing import Any, Sequence

import pytest

from core.agent.evaluator import EvaluationResult
from core.agent.scheduler import StrategyJob, StrategyScheduler
from core.agent.strategy import Strategy


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, delta: float) -> None:
        self._now += float(delta)


class DummyEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[Strategy], Any, bool]] = []

    def evaluate(
        self,
        strategies: Sequence[Strategy],
        data: Any,
        *,
        raise_on_error: bool = False,
    ) -> list[EvaluationResult]:
        self.calls.append((tuple(strategies), data, raise_on_error))
        results: list[EvaluationResult] = []
        for strategy in strategies:
            results.append(EvaluationResult(strategy=strategy, score=1.0, duration=0.0, error=None))
        return results


def _make_scheduler(
    clock: FakeClock,
    evaluator: DummyEvaluator | None = None,
    *,
    max_workers: int | None = None,
) -> StrategyScheduler:
    evaluator = evaluator or DummyEvaluator()
    return StrategyScheduler(
        evaluator=evaluator,
        time_source=clock.now,
        sleep=lambda _: None,
        rng=random.Random(42),
        max_backoff=3600.0,
        max_workers=max_workers,
    )


def test_scheduler_executes_due_jobs() -> None:
    clock = FakeClock()
    evaluator = DummyEvaluator()
    scheduler = _make_scheduler(clock, evaluator)

    strategy = Strategy(name="mean_revert", params={"lookback": 20, "threshold": 0.5})
    data_points = [100.0, 101.5, 102.0, 101.0, 100.5]

    job = StrategyJob(
        name="replay",
        strategies=[strategy],
        data_provider=lambda: data_points,
        interval=5.0,
    )
    scheduler.add_job(job)

    assert scheduler.run_pending() == {}

    clock.advance(5.0)
    results = scheduler.run_pending()
    assert "replay" in results
    replay_results = results["replay"]
    assert len(replay_results) == 1
    assert isinstance(replay_results[0], EvaluationResult)
    assert evaluator.calls[0][0][0] is strategy

    status = scheduler.get_status("replay")
    assert status.result_count == 1
    assert status.last_error is None
    assert status.consecutive_failures == 0
    assert status.last_run_at == pytest.approx(clock.now())


def test_scheduler_respects_jitter_and_reschedules() -> None:
    clock = FakeClock()
    scheduler = _make_scheduler(clock)

    strategy = Strategy(name="momentum", params={"lookback": 15, "threshold": 0.3})
    job = StrategyJob(
        name="jittered",
        strategies=[strategy],
        data_provider=lambda: [1.0, 1.1, 1.2, 1.3],
        interval=10.0,
        jitter=2.0,
    )
    scheduler.add_job(job)

    status = scheduler.get_status("jittered")
    first_delay = status.next_run_at - clock.now()
    assert 8.0 <= first_delay <= 12.0

    clock.advance(first_delay)
    scheduler.run_pending()
    next_status = scheduler.get_status("jittered")
    second_delay = next_status.next_run_at - clock.now()
    assert 8.0 <= second_delay <= 12.0


def test_scheduler_applies_backoff_on_failure() -> None:
    clock = FakeClock()
    evaluator = DummyEvaluator()
    scheduler = _make_scheduler(clock, evaluator)

    strategy = Strategy(name="fail", params={"lookback": 10, "threshold": 0.2})
    def failing_data() -> list[float]:
        raise RuntimeError("dataset unavailable")

    job = StrategyJob(
        name="failing",
        strategies=[strategy],
        data_provider=failing_data,
        interval=4.0,
    )
    scheduler.add_job(job, run_immediately=True)

    scheduler.run_pending()
    status = scheduler.get_status("failing")
    assert status.consecutive_failures == 1
    assert isinstance(status.last_error, RuntimeError)
    assert status.result_count == 0
    first_backoff = status.next_run_at - clock.now()
    assert pytest.approx(first_backoff) == 4.0
    errors = scheduler.drain_failures()
    assert list(errors) == ["failing"]
    assert isinstance(errors["failing"], RuntimeError)
    assert scheduler.drain_failures() == {}

    clock.advance(first_backoff)
    scheduler.run_pending()
    status = scheduler.get_status("failing")
    assert status.consecutive_failures == 2
    second_backoff = status.next_run_at - clock.now()
    assert pytest.approx(second_backoff) == 8.0


def test_scheduler_invokes_callbacks() -> None:
    clock = FakeClock()
    evaluator = DummyEvaluator()
    scheduler = _make_scheduler(clock, evaluator)

    strategy = Strategy(name="callbacks", params={"lookback": 25, "threshold": 0.4})
    completed: list[Sequence[EvaluationResult]] = []
    errors: list[BaseException] = []

    def on_complete(job: StrategyJob, results: Sequence[EvaluationResult]) -> None:
        completed.append(results)

    def on_error(job: StrategyJob, error: BaseException) -> None:
        errors.append(error)

    data_points = [100.0, 101.0, 99.0, 100.5]

    job = StrategyJob(
        name="success",
        strategies=[strategy],
        data_provider=lambda: data_points,
        interval=2.0,
        on_complete=on_complete,
        on_error=on_error,
    )
    scheduler.add_job(job, run_immediately=True)

    scheduler.run_pending()
    assert completed and isinstance(completed[0][0], EvaluationResult)
    assert not errors

    # Force an error and ensure the error callback is invoked.
    def failing_provider() -> Sequence[float]:
        raise RuntimeError("boom")

    job.data_provider = failing_provider
    clock.advance(2.0)
    scheduler.run_pending()
    assert errors and isinstance(errors[0], RuntimeError)


class SlowEvaluator(DummyEvaluator):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def evaluate(
        self,
        strategies: Sequence[Strategy],
        data: Any,
        *,
        raise_on_error: bool = False,
    ) -> list[EvaluationResult]:
        time.sleep(self.delay)
        return super().evaluate(strategies, data, raise_on_error=raise_on_error)


class BlockingEvaluator(DummyEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self._release = threading.Event()

    def evaluate(
        self,
        strategies: Sequence[Strategy],
        data: Any,
        *,
        raise_on_error: bool = False,
    ) -> list[EvaluationResult]:
        self.started.set()
        released = self._release.wait(timeout=1.0)
        if not released:  # pragma: no cover - defensive timeout
            raise TimeoutError("BlockingEvaluator timed out waiting for release")
        return super().evaluate(strategies, data, raise_on_error=raise_on_error)

    def release(self) -> None:
        self._release.set()


def test_scheduler_runs_jobs_in_parallel() -> None:
    delay = 0.2
    evaluator = SlowEvaluator(delay)
    scheduler = StrategyScheduler(
        evaluator=evaluator,
        time_source=time.monotonic,
        sleep=lambda _: None,
        rng=random.Random(1337),
        max_backoff=3600.0,
        max_workers=2,
    )

    strategy_a = Strategy(name="parallel-a", params={"lookback": 10, "threshold": 0.1})
    strategy_b = Strategy(name="parallel-b", params={"lookback": 12, "threshold": 0.2})
    dataset = [100.0, 100.5, 101.0, 100.7]

    job_a = StrategyJob(
        name="job-a",
        strategies=[strategy_a],
        data_provider=lambda: dataset,
        interval=60.0,
    )
    job_b = StrategyJob(
        name="job-b",
        strategies=[strategy_b],
        data_provider=lambda: dataset,
        interval=60.0,
    )

    scheduler.add_job(job_a, run_immediately=True)
    scheduler.add_job(job_b, run_immediately=True)

    start = time.perf_counter()
    results = scheduler.run_pending()
    elapsed = time.perf_counter() - start

    assert set(results) == {"job-a", "job-b"}
    assert elapsed < delay * 1.6  # should complete faster than strict serial execution

    scheduler.shutdown()


def test_run_pending_can_dispatch_without_waiting() -> None:
    clock = FakeClock()
    evaluator = BlockingEvaluator()
    scheduler = _make_scheduler(clock, evaluator, max_workers=1)

    strategy = Strategy(name="async", params={"lookback": 8, "threshold": 0.2})
    dataset = [1.0, 1.1, 1.2]
    job = StrategyJob(
        name="async-job",
        strategies=[strategy],
        data_provider=lambda: dataset,
        interval=10.0,
    )

    scheduler.add_job(job, run_immediately=True)

    start = time.perf_counter()
    results = scheduler.run_pending(wait=False)
    elapsed = time.perf_counter() - start

    assert results == {}
    assert elapsed < 0.05
    assert evaluator.started.wait(timeout=0.1)

    status = scheduler.get_status("async-job")
    assert status.in_flight is True

    evaluator.release()

    for _ in range(50):
        status = scheduler.get_status("async-job")
        if not status.in_flight:
            break
        time.sleep(0.01)
    else:  # pragma: no cover - defensive guard
        pytest.fail("job never completed")

    assert status.result_count == 1
    last_results = scheduler.get_last_results("async-job")
    assert last_results is not None
    drained = scheduler.run_pending()
    assert drained == {"async-job": list(last_results)}
