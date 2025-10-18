from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from observability.finops import (
    Budget,
    FinOpsAlert,
    FinOpsController,
    ResourceUsageSample,
)


class AlertCollector:
    def __init__(self) -> None:
        self.alerts: list[FinOpsAlert] = []

    def handle_alert(self, alert: FinOpsAlert) -> None:
        self.alerts.append(alert)


def _ts(offset_hours: int) -> datetime:
    base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    return base + timedelta(hours=offset_hours)


def test_resource_usage_sample_validation() -> None:
    timestamp = _ts(0)
    sample = ResourceUsageSample(
        resource_id="worker-a",
        timestamp=timestamp,
        cost=12.5,
        usage={"cpu_utilisation": 0.42},
    )
    assert sample.resource_id == "worker-a"
    assert sample.timestamp == timestamp

    with pytest.raises(ValueError):
        ResourceUsageSample(resource_id="", timestamp=timestamp, cost=0.0)
    with pytest.raises(ValueError):
        ResourceUsageSample(resource_id="x", timestamp=timestamp, cost=-1.0)
    with pytest.raises(ValueError):
        ResourceUsageSample(
            resource_id="x", timestamp=timestamp, cost=1.0, usage={"cpu": -0.5}
        )


def test_budget_alerts_trigger_and_reset() -> None:
    collector = AlertCollector()
    controller = FinOpsController(alert_sink=collector)
    budget = Budget(
        name="prod-core",
        limit=100.0,
        period=timedelta(hours=6),
        scope={"env": "prod"},
        alert_thresholds=(0.5, 0.8, 1.0),
    )
    controller.add_budget(budget)

    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-1",
            timestamp=_ts(0),
            cost=40.0,
            usage={"cpu_utilisation": 0.55},
            metadata={"env": "prod"},
        )
    )
    assert collector.alerts == []

    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-2",
            timestamp=_ts(1),
            cost=15.0,
            usage={"cpu_utilisation": 0.61},
            metadata={"env": "prod"},
        )
    )
    assert [alert.threshold for alert in collector.alerts] == [0.5]

    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-3",
            timestamp=_ts(2),
            cost=30.0,
            usage={"cpu_utilisation": 0.6},
            metadata={"env": "prod"},
        )
    )
    thresholds = [alert.threshold for alert in collector.alerts]
    assert thresholds == [0.5, 0.8]

    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-3",
            timestamp=_ts(3),
            cost=25.0,
            usage={"cpu_utilisation": 0.7},
            metadata={"env": "prod"},
        )
    )
    thresholds = [alert.threshold for alert in collector.alerts]
    assert thresholds == [0.5, 0.8, 1.0]
    assert collector.alerts[-1].breached is True

    # Usage falls below minimum threshold causing reset
    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-1",
            timestamp=_ts(12),
            cost=5.0,
            usage={"cpu_utilisation": 0.1},
            metadata={"env": "prod"},
        )
    )
    collector.alerts.clear()
    controller.record_usage(
        ResourceUsageSample(
            resource_id="node-1",
            timestamp=_ts(13),
            cost=55.0,
            usage={"cpu_utilisation": 0.55},
            metadata={"env": "prod"},
        )
    )
    assert [alert.threshold for alert in collector.alerts] == [0.5]


def test_analyse_costs_and_recommendations() -> None:
    controller = FinOpsController()
    for idx in range(4):
        controller.record_usage(
            ResourceUsageSample(
                resource_id="db-primary",
                timestamp=_ts(idx),
                cost=35.0,
                usage={"cpu_utilisation": 0.12, "storage_gb": 200.0},
                metadata={"env": "prod"},
            )
        )

    controller.record_usage(
        ResourceUsageSample(
            resource_id="db-primary",
            timestamp=_ts(4),
            cost=90.0,
            usage={"cpu_utilisation": 0.18, "storage_gb": 200.0},
            metadata={"env": "prod"},
        )
    )

    report = controller.analyse_costs(
        timedelta(hours=6), as_of=_ts(5), metadata_filter={"env": "prod"}
    )
    assert pytest.approx(report.total_cost, rel=1e-6) == 230.0
    assert report.resource_costs == {"db-primary": 230.0}
    assert report.usage_totals["storage_gb"] == pytest.approx(1000.0)

    recommendations = controller.recommend_optimisations(
        timedelta(hours=6),
        as_of=_ts(5),
        metadata_filter={"env": "prod"},
    )
    assert any(
        rec for rec in recommendations if "rightsizing" in rec.message
    ), "Expected rightsizing recommendation"
    assert any(
        rec for rec in recommendations if "spikes" in rec.message
    ), "Expected spike investigation recommendation"

