# SPDX-License-Identifier: MIT
"""SLO guardrails and automated rollback helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Callable, Deque, Dict, Iterable, Optional


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SLOConfig:
    """Configuration for evaluating service level objectives.

    Attributes:
        error_rate_threshold: Maximum acceptable error rate expressed as a
            fraction between 0 and 1.
        latency_threshold_ms: Maximum acceptable p95 latency in milliseconds.
        evaluation_period: Sliding window used when evaluating the SLO. Only
            samples newer than ``now - evaluation_period`` are considered.
        min_requests: Minimum amount of samples required before the SLO can be
            evaluated. This avoids triggering on sparse data.
        cooldown: Minimum amount of time that needs to elapse between
            consecutive rollbacks. Prevents flapping when the system is already
            in mitigation mode.
    """

    error_rate_threshold: float = 0.02
    latency_threshold_ms: float = 500.0
    evaluation_period: timedelta = timedelta(minutes=5)
    min_requests: int = 50
    cooldown: timedelta = timedelta(minutes=5)


@dataclass
class RequestSample:
    """Individual request sample used for sliding-window evaluations."""

    timestamp: datetime
    latency_ms: float
    success: bool


class AutoRollbackGuard:
    """Evaluate SLO windows and trigger rollback callbacks when breached."""

    def __init__(
        self,
        config: SLOConfig | None = None,
        *,
        rollback_callback: Optional[Callable[[str, Dict[str, float]], None]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config or SLOConfig()
        self._rollback_callback = rollback_callback
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: Deque[RequestSample] = deque()
        self._last_triggered_at: Optional[datetime] = None
        self._last_summary: Dict[str, float] | None = None

    @property
    def last_triggered_at(self) -> Optional[datetime]:
        """Return the timestamp of the last rollback trigger."""

        return self._last_triggered_at

    @property
    def last_summary(self) -> Dict[str, float] | None:
        """Return the metrics snapshot from the last evaluation."""

        return self._last_summary.copy() if self._last_summary is not None else None

    def record_outcome(
        self,
        latency_ms: float,
        success: bool,
        *,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record an individual request outcome.

        Args:
            latency_ms: Request latency in milliseconds.
            success: ``True`` for successful requests, ``False`` otherwise.
            timestamp: Optional timestamp (defaults to ``datetime.now``).

        Returns:
            ``True`` when a rollback should be triggered as the result of this
            sample, ``False`` otherwise.
        """

        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        event_time = timestamp or self._clock()
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)

        self._events.append(RequestSample(event_time, float(latency_ms), success))
        self._prune(event_time)
        summary = self._summarise_window()
        if summary is None:
            return False

        reason = self._breach_reason(summary)
        if reason is None:
            self._last_summary = summary
            return False

        return self._trigger(reason, summary, event_time)

    def evaluate_snapshot(
        self,
        *,
        error_rate: float,
        latency_p95_ms: float,
        timestamp: Optional[datetime] = None,
        total_requests: Optional[int] = None,
    ) -> bool:
        """Evaluate the SLO guard using pre-aggregated metrics.

        This helper is useful when metrics are sourced from an external system
        such as Prometheus or Datadog and the agent receives aggregated error
        rate and latency data instead of individual request samples.
        """

        if latency_p95_ms < 0:
            raise ValueError("latency_p95_ms must be non-negative")
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")

        now = timestamp or self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        summary: Dict[str, float] = {
            "error_rate": float(error_rate),
            "latency_p95_ms": float(latency_p95_ms),
            "total_requests": float(total_requests) if total_requests is not None else math.nan,
            "window_seconds": self.config.evaluation_period.total_seconds(),
        }

        reason = self._breach_reason(summary)
        if reason is None:
            self._last_summary = summary
            return False

        return self._trigger(reason, summary, now)

    def _breach_reason(self, summary: Dict[str, float]) -> Optional[str]:
        error_rate = summary.get("error_rate", 0.0)
        latency_p95 = summary.get("latency_p95_ms", 0.0)
        if error_rate >= self.config.error_rate_threshold:
            return "error_rate"
        if latency_p95 >= self.config.latency_threshold_ms:
            return "latency"
        return None

    def _trigger(self, reason: str, summary: Dict[str, float], now: datetime) -> bool:
        if self._last_triggered_at is not None:
            elapsed = now - self._last_triggered_at
            if elapsed < self.config.cooldown:
                self._last_summary = summary
                return False

        self._last_triggered_at = now
        enriched_summary = dict(summary)
        enriched_summary.update({
            "reason": reason,
            "triggered_at": now.timestamp(),
            "cooldown_seconds": self.config.cooldown.total_seconds(),
        })
        self._last_summary = enriched_summary

        _logger.warning(
            "SLO breach detected — initiating rollback",
            extra={
                "reason": reason,
                "error_rate": enriched_summary.get("error_rate"),
                "latency_p95_ms": enriched_summary.get("latency_p95_ms"),
                "total_requests": enriched_summary.get("total_requests"),
            },
        )

        if self._rollback_callback is not None:
            self._rollback_callback(reason, enriched_summary)
        return True

    def _summarise_window(self) -> Optional[Dict[str, float]]:
        total = len(self._events)
        if total < self.config.min_requests:
            return None

        errors = sum(1 for event in self._events if not event.success)
        latencies = [event.latency_ms for event in self._events]
        error_rate = errors / total if total else 0.0
        latency_p95 = _percentile(latencies, 95.0)
        return {
            "error_rate": error_rate,
            "latency_p95_ms": latency_p95,
            "total_requests": float(total),
            "window_seconds": self.config.evaluation_period.total_seconds(),
        }

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.config.evaluation_period
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()


def _percentile(values: Iterable[float], percentile: float) -> float:
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if percentile <= 0:
        return data[0]
    if percentile >= 100:
        return data[-1]

    rank = (percentile / 100) * (len(data) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return data[int(rank)]

    fraction = rank - lower
    return data[lower] + (data[upper] - data[lower]) * fraction
