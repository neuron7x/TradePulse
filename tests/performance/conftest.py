"""Pytest fixtures and utilities for performance regression enforcement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest

_BASELINES_PATH = Path(__file__).with_name("benchmark_baselines.json")
_DEFAULT_THRESHOLD = 0.15  # Allow up to +15% regression by default.
_ARTIFACT_DIR = Path(os.environ.get("TP_BENCHMARK_ARTIFACT_DIR", "reports/benchmarks"))

try:
    _ARTIFACT_TTL_DAYS = int(os.environ.get("TP_BENCHMARK_ARTIFACT_TTL_DAYS", "14"))
except ValueError:  # pragma: no cover - defensive configuration guard
    _ARTIFACT_TTL_DAYS = 14


@dataclass(frozen=True, slots=True)
class _BenchmarkRecord:
    """Container tracking a single benchmark measurement."""

    test_id: str
    name: str
    mean: float
    baseline: float
    threshold: float

    @property
    def regression_ratio(self) -> float:
        return self.mean / self.baseline if self.baseline else float("inf")

    @property
    def exceeds_budget(self) -> bool:
        return self.regression_ratio > 1.0 + self.threshold


class _BenchmarkMonitor:
    """Session-level registry to summarise benchmark regressions."""

    def __init__(self) -> None:
        self.records: list[_BenchmarkRecord] = []

    def add(self, record: _BenchmarkRecord) -> None:
        self.records.append(record)

    def iter_failures(self) -> list[_BenchmarkRecord]:
        return [record for record in self.records if record.exceeds_budget]


_monitor = _BenchmarkMonitor()


def _load_baselines() -> dict[str, float]:
    try:
        data = json.loads(_BASELINES_PATH.read_text())
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(
            "Benchmark baseline file is missing. Generate baselines before running performance tests."
        ) from exc
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise RuntimeError(
            "Benchmark baseline file must contain a mapping of benchmark keys to floats."
        )
    return {str(key): float(value) for key, value in data.items()}


@pytest.fixture(scope="session")
def benchmark_baselines() -> dict[str, float]:
    """Return benchmark baseline timings in seconds."""

    return _load_baselines()


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.fixture
def benchmark_guard(
    request: pytest.FixtureRequest,
    benchmark_baselines: dict[str, float],
) -> Callable[..., Any]:
    """Wrap :func:`pytest_benchmark` to enforce regression budgets.

    The fixture attempts to retrieve the :mod:`pytest-benchmark` integration at
    runtime and will mark the requesting test as skipped when the plugin is not
    installed. This avoids hard errors in developer environments that do not
    have benchmarking dependencies available by default.

    Parameters
    ----------
    request:
        Current pytest request object (used to report test id).
    benchmark_baselines:
        Mapping of benchmark keys to baseline mean runtimes in seconds.
    """

    try:
        benchmark: "BenchmarkFixture" = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("pytest-benchmark plugin is required for performance tests")

    def _runner(
        func: Callable[..., Any],
        *args: Any,
        baseline_key: str,
        threshold: float = _DEFAULT_THRESHOLD,
        rounds: int = 8,
        warmup_rounds: int = 1,
        **kwargs: Any,
    ) -> Any:
        if baseline_key not in benchmark_baselines:
            raise KeyError(
                f"Benchmark baseline '{baseline_key}' is missing. "
                "Update tests/performance/benchmark_baselines.json with a reference measurement."
            )

        result = benchmark.pedantic(
            func,
            args=args,
            kwargs=kwargs,
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )

        stats = getattr(benchmark, "stats", None)
        if stats is None or not hasattr(stats, "stats"):
            raise RuntimeError(
                "pytest-benchmark did not provide timing statistics for the executed benchmark"
            )
        mean = float(stats["mean"])

        baseline = benchmark_baselines[baseline_key]
        record = _BenchmarkRecord(
            test_id=request.node.nodeid,
            name=baseline_key,
            mean=mean,
            baseline=baseline,
            threshold=threshold,
        )
        _monitor.add(record)

        if record.exceeds_budget:
            raise AssertionError(
                (
                    f"Benchmark '{baseline_key}' regressed: mean {mean:.6f}s vs baseline {baseline:.6f}s "
                    f"(+{(record.regression_ratio - 1.0):.1%} > allowed {threshold:.1%})."
                )
            )

        return result

    return _runner


def _serialize_record(record: _BenchmarkRecord) -> dict[str, Any]:
    return {
        "test_id": record.test_id,
        "name": record.name,
        "mean": record.mean,
        "baseline": record.baseline,
        "threshold": record.threshold,
        "regression_ratio": record.regression_ratio,
        "exceeds_budget": record.exceeds_budget,
    }


def _persist_benchmark_artifacts(records: list[_BenchmarkRecord]) -> None:
    if not records:
        return

    timestamp = datetime.now(timezone.utc)
    artifact_dir = _ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": timestamp.isoformat(),
        "records": [_serialize_record(record) for record in records],
    }
    artifact_path = (
        artifact_dir / f"benchmark-summary-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if _ARTIFACT_TTL_DAYS <= 0:
        return

    cutoff = timestamp - timedelta(days=_ARTIFACT_TTL_DAYS)
    for existing in artifact_dir.glob("benchmark-summary-*.json"):
        try:
            if (
                datetime.fromtimestamp(existing.stat().st_mtime, tz=timezone.utc)
                < cutoff
            ):
                existing.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            continue


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int) -> None:
    """Emit a concise summary for CI logs detailing benchmark deviations."""

    if not _monitor.records:
        return

    terminalreporter.write_sep("=", "benchmark regression summary")
    header = f"{'benchmark':40} {'mean (s)':>12} {'baseline (s)':>14} {'delta':>9} {'budget':>8}"
    terminalreporter.write_line(header)
    for record in _monitor.records:
        delta_pct = (record.regression_ratio - 1.0) * 100.0
        status = "FAIL" if record.exceeds_budget else "OK"
        terminalreporter.write_line(
            f"{record.name:40} {record.mean:12.6f} {record.baseline:14.6f} {delta_pct:8.2f}% {status:>8}"
        )

    failures = _monitor.iter_failures()
    if failures:
        terminalreporter.write_sep("-", "benchmark regressions detected")
        for record in failures:
            terminalreporter.write_line(
                f"{record.test_id}: {record.name} exceeded threshold by {(record.regression_ratio - 1.0):.1%}"
            )

    _persist_benchmark_artifacts(_monitor.records)
