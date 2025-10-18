"""FinOps cost control utilities for TradePulse.

The module provides primitives to monitor infrastructure usage, evaluate budget
compliance, and emit actionable optimisation guidance.  It favours an
in-memory ledger that can be embedded in schedulers, CLI tooling, or
server-side daemons without requiring a backing database.  Consumers feed
`ResourceUsageSample` instances obtained from billing exports or provider APIs
and the controller keeps rolling aggregates aligned with configured budgets.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import statistics
from typing import (
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .notifications import NotificationDispatcher

__all__ = [
    "ResourceUsageSample",
    "Budget",
    "BudgetStatus",
    "CostReport",
    "OptimizationRecommendation",
    "FinOpsAlert",
    "AlertSink",
    "NotificationAlertSink",
    "FinOpsController",
]


def _ensure_positive_duration(value: timedelta) -> None:
    if value <= timedelta(0):
        raise ValueError("Duration must be positive")


@dataclass(slots=True, frozen=True)
class ResourceUsageSample:
    """Snapshot describing the utilisation and spend of a resource."""

    resource_id: str
    timestamp: datetime
    cost: float
    usage: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource_id must be provided")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")
        if self.cost < 0.0 or not math.isfinite(self.cost):
            raise ValueError("cost must be a finite, non-negative number")
        for key, value in self.usage.items():
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(
                    f"usage metric '{key}' must be a finite, non-negative number"
                )


@dataclass(slots=True, frozen=True)
class Budget:
    """Declarative budget with optional metadata matching scope."""

    name: str
    limit: float
    period: timedelta
    scope: Mapping[str, str] | None = None
    alert_thresholds: Sequence[float] = (0.8, 1.0)
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Budget name must be provided")
        if self.limit <= 0.0 or not math.isfinite(self.limit):
            raise ValueError("Budget limit must be a finite, positive number")
        _ensure_positive_duration(self.period)
        if not self.alert_thresholds:
            raise ValueError("At least one alert threshold must be specified")
        sorted_thresholds = sorted(self.alert_thresholds)
        if any(threshold <= 0.0 for threshold in sorted_thresholds):
            raise ValueError("Alert thresholds must be greater than zero")
        if any(not math.isfinite(threshold) for threshold in sorted_thresholds):
            raise ValueError("Alert thresholds must be finite")
        object.__setattr__(self, "alert_thresholds", tuple(sorted_thresholds))


@dataclass(slots=True, frozen=True)
class BudgetStatus:
    """Current utilisation snapshot of a budget."""

    budget: Budget
    total_cost: float
    window_start: datetime
    window_end: datetime
    utilisation: float
    remaining: float
    breached: bool


@dataclass(slots=True, frozen=True)
class CostReport:
    """Aggregated cost analytics for a time window."""

    total_cost: float
    average_daily_cost: float
    max_sample_cost: float
    resource_costs: Mapping[str, float]
    usage_totals: Mapping[str, float]
    window_start: datetime
    window_end: datetime


@dataclass(slots=True, frozen=True)
class OptimizationRecommendation:
    """Actionable optimisation recommendation derived from usage data."""

    resource_id: str
    message: str
    severity: str
    metadata: Mapping[str, float | str]


@dataclass(slots=True, frozen=True)
class FinOpsAlert:
    """Alert emitted when a budget crosses a utilisation threshold."""

    budget: Budget
    threshold: float
    total_cost: float
    remaining: float
    utilisation: float
    window_start: datetime
    window_end: datetime
    breached: bool


@runtime_checkable
class AlertSink(Protocol):
    """Protocol implemented by alert sinks receiving FinOps alerts."""

    def handle_alert(self, alert: FinOpsAlert) -> None:  # pragma: no cover - protocol
        """Process a triggered alert."""


class NotificationAlertSink:
    """Adapter that routes alerts through :class:`NotificationDispatcher`."""

    def __init__(
        self,
        dispatcher: NotificationDispatcher,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._loop = loop

    def handle_alert(self, alert: FinOpsAlert) -> None:
        subject, message, metadata = self._build_payload(alert)
        coroutine = self._dispatcher.dispatch(
            "finops.budget", subject=subject, message=message, metadata=metadata
        )
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(coroutine)
                return
        if loop.is_running():
            loop.create_task(coroutine)
        else:  # pragma: no cover - exercised when explicitly providing event loop
            loop.run_until_complete(coroutine)

    @staticmethod
    def _build_payload(alert: FinOpsAlert) -> tuple[str, str, Mapping[str, object]]:
        budget = alert.budget
        utilisation_pct = round(alert.utilisation * 100, 2)
        status = "breached" if alert.breached else "warning"
        subject = f"[FinOps] Budget {budget.name} {status}"
        message = (
            f"Budget '{budget.name}' is at {utilisation_pct}% of its limit"
            f" ({budget.currency} {alert.total_cost:.2f} spent of {budget.currency} {budget.limit:.2f})."
        )
        if alert.breached:
            message += " Limit exceeded; immediate action required."
        metadata = {
            "budget": budget.name,
            "threshold_ratio": round(alert.threshold, 3),
            "utilisation_pct": utilisation_pct,
            "total_cost": round(alert.total_cost, 2),
            "remaining_budget": round(alert.remaining, 2),
            "window_start": alert.window_start.isoformat(),
            "window_end": alert.window_end.isoformat(),
            "currency": budget.currency,
        }
        if budget.scope:
            metadata["scope"] = dict(budget.scope)
        return subject, message, metadata


class FinOpsController:
    """Coordinate FinOps analytics, budgets, and alerting."""

    def __init__(
        self,
        *,
        alert_sink: AlertSink | None = None,
        clock: callable[[], datetime] | None = None,
    ) -> None:
        self._alert_sink = alert_sink
        self._clock = clock or datetime.utcnow
        self._usage_by_resource: MutableMapping[str, list[ResourceUsageSample]] = (
            defaultdict(list)
        )
        self._budgets: MutableMapping[str, Budget] = {}
        self._budget_ledgers: MutableMapping[
            str, list[tuple[datetime, float]]
        ] = defaultdict(list)
        self._budget_thresholds: MutableMapping[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------
    def add_budget(self, budget: Budget) -> None:
        """Register a new budget, replacing any previous definition."""

        self._budgets[budget.name] = budget
        self._budget_ledgers.setdefault(budget.name, [])
        self._budget_thresholds.setdefault(budget.name, 0.0)

    def remove_budget(self, name: str) -> None:
        """Remove a configured budget."""

        self._budgets.pop(name, None)
        self._budget_ledgers.pop(name, None)
        self._budget_thresholds.pop(name, None)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def record_usage(self, sample: ResourceUsageSample) -> tuple[FinOpsAlert, ...]:
        """Record a usage sample and evaluate affected budgets."""

        self._store_usage_sample(sample)
        alerts: list[FinOpsAlert] = []
        for budget in self._budgets.values():
            if not self._sample_matches_scope(sample, budget.scope):
                continue
            ledger = self._budget_ledgers[budget.name]
            self._insert_ledger_entry(ledger, sample)
            total_cost, window_start = self._prune_and_sum(ledger, budget, sample.timestamp)
            alert_events = self._evaluate_budget(
                budget,
                total_cost,
                window_start,
                sample.timestamp,
            )
            if alert_events:
                alerts.extend(alert_events)
        return tuple(alerts)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def analyse_costs(
        self,
        window: timedelta,
        *,
        metadata_filter: Mapping[str, str] | None = None,
        as_of: datetime | None = None,
    ) -> CostReport:
        """Aggregate cost and usage statistics across resources."""

        _ensure_positive_duration(window)
        as_of = as_of or self._clock()
        window_start = as_of - window

        resource_costs: MutableMapping[str, float] = defaultdict(float)
        usage_totals: MutableMapping[str, float] = defaultdict(float)
        total_cost = 0.0
        max_sample_cost = 0.0

        for resource_id, samples in self._usage_by_resource.items():
            relevant_samples = self._slice_samples(
                samples, window_start, as_of, metadata_filter
            )
            resource_sum = 0.0
            for sample in relevant_samples:
                resource_sum += sample.cost
                total_cost += sample.cost
                max_sample_cost = max(max_sample_cost, sample.cost)
                for metric, value in sample.usage.items():
                    usage_totals[metric] += value
            if resource_sum:
                resource_costs[resource_id] += resource_sum

        days = max(window.total_seconds() / 86400.0, 1e-6)
        average_daily_cost = total_cost / days

        return CostReport(
            total_cost=total_cost,
            average_daily_cost=average_daily_cost,
            max_sample_cost=max_sample_cost,
            resource_costs=dict(sorted(resource_costs.items())),
            usage_totals=dict(sorted(usage_totals.items())),
            window_start=window_start,
            window_end=as_of,
        )

    def recommend_optimisations(
        self,
        window: timedelta,
        *,
        utilisation_threshold: float = 0.25,
        spike_multiplier: float = 1.5,
        metadata_filter: Mapping[str, str] | None = None,
        as_of: datetime | None = None,
    ) -> tuple[OptimizationRecommendation, ...]:
        """Derive optimisation recommendations from recent usage."""

        if utilisation_threshold <= 0.0:
            raise ValueError("utilisation_threshold must be positive")
        if spike_multiplier <= 1.0:
            raise ValueError("spike_multiplier must be greater than 1")

        _ensure_positive_duration(window)
        as_of = as_of or self._clock()
        window_start = as_of - window

        recommendations: list[OptimizationRecommendation] = []

        for resource_id, samples in self._usage_by_resource.items():
            relevant = self._slice_samples(samples, window_start, as_of, metadata_filter)
            if not relevant:
                continue

            utilisation_values: list[float] = []
            costs = [sample.cost for sample in relevant]
            for sample in relevant:
                for value in sample.usage.values():
                    if 0.0 <= value <= 1.0 and math.isfinite(value):
                        utilisation_values.append(value)

            if utilisation_values:
                avg_utilisation = statistics.fmean(utilisation_values)
                total_cost = sum(costs)
                if avg_utilisation < utilisation_threshold and total_cost > 0.0:
                    recommendations.append(
                        OptimizationRecommendation(
                            resource_id=resource_id,
                            message=(
                                "Resource exhibits sustained low utilisation; consider rightsizing or scheduling shutdowns."
                            ),
                            severity="medium",
                            metadata={
                                "average_utilisation": round(avg_utilisation, 4),
                                "total_cost": round(total_cost, 2),
                            },
                        )
                    )

            if len(costs) >= 3:
                historical = costs[:-1]
                latest = costs[-1]
                median_cost = statistics.median(historical)
                if median_cost > 0.0 and latest >= median_cost * spike_multiplier:
                    recommendations.append(
                        OptimizationRecommendation(
                            resource_id=resource_id,
                            message="Latest cost sample spikes above historical trend; investigate anomalies or misconfigurations.",
                            severity="high",
                            metadata={
                                "latest_cost": round(latest, 2),
                                "median_cost": round(median_cost, 2),
                                "spike_multiplier": spike_multiplier,
                            },
                        )
                    )

        return tuple(recommendations)

    # ------------------------------------------------------------------
    # Budget status queries
    # ------------------------------------------------------------------
    def get_budget_status(
        self, name: str, *, as_of: datetime | None = None
    ) -> BudgetStatus:
        """Return the current status of a budget."""

        if name not in self._budgets:
            raise KeyError(f"Unknown budget '{name}'")
        budget = self._budgets[name]
        as_of = as_of or self._clock()
        ledger = self._budget_ledgers.get(name, [])
        total_cost, window_start = self._prune_and_sum(ledger, budget, as_of)
        utilisation = total_cost / budget.limit
        remaining = max(budget.limit - total_cost, 0.0)
        return BudgetStatus(
            budget=budget,
            total_cost=total_cost,
            window_start=window_start,
            window_end=as_of,
            utilisation=utilisation,
            remaining=remaining,
            breached=total_cost >= budget.limit,
        )

    def iter_budget_statuses(
        self, *, as_of: datetime | None = None
    ) -> Sequence[BudgetStatus]:
        """Return statuses for all budgets."""

        as_of = as_of or self._clock()
        return [self.get_budget_status(name, as_of=as_of) for name in self._budgets]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _store_usage_sample(self, sample: ResourceUsageSample) -> None:
        entries = self._usage_by_resource[sample.resource_id]
        if not entries or sample.timestamp >= entries[-1].timestamp:
            entries.append(sample)
            return
        timestamps = [entry.timestamp for entry in entries]
        index = bisect_left(timestamps, sample.timestamp)
        entries.insert(index, sample)

    def _sample_matches_scope(
        self, sample: ResourceUsageSample, scope: Mapping[str, str] | None
    ) -> bool:
        if not scope:
            return True
        sample_meta = sample.metadata
        return all(sample_meta.get(key) == value for key, value in scope.items())

    def _insert_ledger_entry(
        self, ledger: list[tuple[datetime, float]], sample: ResourceUsageSample
    ) -> None:
        if not ledger or sample.timestamp >= ledger[-1][0]:
            ledger.append((sample.timestamp, sample.cost))
            return
        timestamps = [entry[0] for entry in ledger]
        index = bisect_left(timestamps, sample.timestamp)
        ledger.insert(index, (sample.timestamp, sample.cost))

    def _prune_and_sum(
        self,
        ledger: list[tuple[datetime, float]],
        budget: Budget,
        as_of: datetime,
    ) -> tuple[float, datetime]:
        cutoff = as_of - budget.period
        while ledger and ledger[0][0] < cutoff:
            ledger.pop(0)
        total_cost = sum(cost for _, cost in ledger)
        window_start = cutoff if ledger else as_of - budget.period
        return total_cost, window_start

    def _evaluate_budget(
        self,
        budget: Budget,
        total_cost: float,
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[FinOpsAlert, ...]:
        limit = budget.limit
        utilisation = total_cost / limit if limit else 0.0
        remaining = max(limit - total_cost, 0.0)
        thresholds = budget.alert_thresholds
        current_state = self._budget_thresholds.get(budget.name, 0.0)
        triggered: list[FinOpsAlert] = []

        for threshold in thresholds:
            threshold_cost = limit * threshold
            if total_cost >= threshold_cost and threshold > current_state:
                triggered.append(
                    FinOpsAlert(
                        budget=budget,
                        threshold=threshold,
                        total_cost=total_cost,
                        remaining=remaining,
                        utilisation=utilisation,
                        window_start=window_start,
                        window_end=window_end,
                        breached=threshold >= 1.0 or total_cost >= limit,
                    )
                )
                current_state = max(current_state, threshold)

        if total_cost >= limit and current_state < 1.0:
            triggered.append(
                FinOpsAlert(
                    budget=budget,
                    threshold=1.0,
                    total_cost=total_cost,
                    remaining=remaining,
                    utilisation=utilisation,
                    window_start=window_start,
                    window_end=window_end,
                    breached=True,
                )
            )
            current_state = 1.0

        if triggered:
            self._budget_thresholds[budget.name] = current_state
            for alert in triggered:
                if self._alert_sink is not None:
                    self._alert_sink.handle_alert(alert)
            return tuple(triggered)

        lowest_threshold = thresholds[0] if thresholds else 1.0
        if total_cost < limit * lowest_threshold * 0.8:
            self._budget_thresholds[budget.name] = 0.0
        return ()

    def _slice_samples(
        self,
        samples: Sequence[ResourceUsageSample],
        start: datetime,
        end: datetime,
        metadata_filter: Mapping[str, str] | None,
    ) -> list[ResourceUsageSample]:
        result: list[ResourceUsageSample] = []
        for sample in samples:
            if sample.timestamp < start or sample.timestamp > end:
                continue
            if metadata_filter and not self._sample_matches_scope(sample, metadata_filter):
                continue
            result.append(sample)
        return result

