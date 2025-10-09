# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import math
import pytest

from core.utils.slo import AutoRollbackGuard, SLOConfig


def _ts(offset_seconds: float) -> datetime:
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds)


def test_auto_rollback_triggers_on_error_rate_and_respects_cooldown() -> None:
    triggered: list[tuple[str, dict[str, float]]] = []
    config = SLOConfig(
        error_rate_threshold=0.4,
        latency_threshold_ms=5000.0,
        evaluation_period=timedelta(seconds=30),
        min_requests=5,
        cooldown=timedelta(seconds=20),
    )
    guard = AutoRollbackGuard(
        config,
        rollback_callback=lambda reason, summary: triggered.append((reason, summary)),
    )

    outcomes = [
        (100.0, True),
        (120.0, True),
        (110.0, False),
        (105.0, False),
        (115.0, True),
    ]
    for idx, (latency, success) in enumerate(outcomes):
        guard.record_outcome(latency, success, timestamp=_ts(idx * 2))

    assert triggered and triggered[0][0] == "error_rate"
    summary = triggered[0][1]
    assert pytest.approx(summary["error_rate"], rel=1e-5) == 0.4
    assert summary["reason"] == "error_rate"

    guard.record_outcome(130.0, False, timestamp=_ts(25))
    assert len(triggered) == 1  # Cooldown active

    follow_up = [
        (130.0, True),
        (140.0, False),
        (150.0, False),
        (135.0, True),
        (138.0, True),
    ]
    for idx, (latency, success) in enumerate(follow_up, start=40):
        guard.record_outcome(latency, success, timestamp=_ts(idx))

    assert len(triggered) == 2
    assert triggered[1][0] == "error_rate"


def test_auto_rollback_triggers_on_latency_breach() -> None:
    triggered: list[str] = []
    config = SLOConfig(
        error_rate_threshold=0.9,
        latency_threshold_ms=200.0,
        evaluation_period=timedelta(seconds=10),
        min_requests=3,
        cooldown=timedelta(seconds=1),
    )
    guard = AutoRollbackGuard(
        config,
        rollback_callback=lambda reason, summary: triggered.append(summary["reason"]),
    )

    guard.record_outcome(150.0, True, timestamp=_ts(0))
    guard.record_outcome(250.0, True, timestamp=_ts(2))
    guard.record_outcome(275.0, True, timestamp=_ts(4))

    assert triggered == ["latency"]


def test_evaluate_snapshot_allows_external_metrics() -> None:
    triggered: list[str] = []
    guard = AutoRollbackGuard(
        SLOConfig(error_rate_threshold=0.2, latency_threshold_ms=400.0),
        rollback_callback=lambda reason, summary: triggered.append(reason),
    )

    now = _ts(0)
    should_trigger = guard.evaluate_snapshot(
        error_rate=0.5,
        latency_p95_ms=120.0,
        timestamp=now,
        total_requests=200,
    )
    assert should_trigger is True
    assert triggered == ["error_rate"]

    should_trigger = guard.evaluate_snapshot(
        error_rate=0.1,
        latency_p95_ms=800.0,
        timestamp=now + timedelta(minutes=10),
        total_requests=120,
    )
    assert should_trigger is True
    assert triggered[-1] == "latency"


def test_input_validation() -> None:
    guard = AutoRollbackGuard(SLOConfig())
    with pytest.raises(ValueError):
        guard.record_outcome(latency_ms=-1, success=True, timestamp=_ts(0))
    with pytest.raises(ValueError):
        guard.evaluate_snapshot(error_rate=-0.1, latency_p95_ms=1)
    with pytest.raises(ValueError):
        guard.evaluate_snapshot(error_rate=0.5, latency_p95_ms=-5)


def test_percentile_computation_matches_expected_values() -> None:
    guard = AutoRollbackGuard(SLOConfig(min_requests=1))
    latencies = [10, 20, 30, 40, 50]
    for idx, latency in enumerate(latencies):
        guard.record_outcome(latency, True, timestamp=_ts(idx))
    summary = guard.last_summary
    assert summary is not None
    assert math.isclose(summary["latency_p95_ms"], 48.0)
