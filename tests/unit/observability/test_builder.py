from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.observability.builder import (
    ObservabilityConfigError,
    build_bundle,
    validate_alerts,
    validate_metrics,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_metrics_requires_unique_names(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    _write_json(
        metrics_path,
        {
            "metrics": [
                {
                    "name": "tradepulse_metric_one",
                    "type": "counter",
                    "description": "A metric",
                    "labels": ["label"],
                    "subsystem": "demo",
                },
                {
                    "name": "tradepulse_metric_one",
                    "type": "gauge",
                    "description": "Duplicate",
                    "labels": [],
                    "subsystem": "demo",
                },
            ]
        },
    )

    with pytest.raises(ObservabilityConfigError, match="Duplicate metric definition"):
        validate_metrics(metrics_path)


def test_validate_alerts_rejects_unknown_metric(tmp_path: Path) -> None:
    alerts_path = tmp_path / "alerts.json"
    _write_json(
        alerts_path,
        {
            "groups": [
                {
                    "name": "demo",
                    "rules": [
                        {
                            "alert": "DemoAlert",
                            "expr": "sum(tradepulse_unknown_total)",
                            "labels": {},
                            "annotations": {},
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(ObservabilityConfigError, match="unknown metrics"):
        validate_alerts(alerts_path, ["tradepulse_known_total"])


def test_build_bundle_generates_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    dashboards = root / "dashboards"
    dashboards.mkdir()

    _write_json(
        root / "metrics.json",
        {
            "metrics": [
                {
                    "name": "tradepulse_metric_one",
                    "type": "counter",
                    "description": "A metric",
                    "labels": [],
                    "subsystem": "demo",
                },
                {
                    "name": "tradepulse_metric_two",
                    "type": "histogram",
                    "description": "Another metric",
                    "labels": ["label"],
                    "subsystem": "demo",
                },
            ]
        },
    )

    _write_json(
        root / "alerts.json",
        {
            "groups": [
                {
                    "name": "demo",
                    "rules": [
                        {
                            "alert": "DemoAlert",
                            "expr": "sum(rate(tradepulse_metric_one[5m])) > 10",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "Demo"},
                        }
                    ],
                }
            ]
        },
    )

    _write_json(
        dashboards / "demo.json",
        {
            "uid": "demo",
            "title": "Demo",
            "panels": [
                {
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "sum(tradepulse_metric_one)",
                            "refId": "A",
                        }
                    ],
                }
            ],
        },
    )

    output_dir = root / "generated"
    manifest = build_bundle(root, output_dir)

    alerts_file = output_dir / "prometheus" / "alerts.yaml"
    dashboard_file = output_dir / "dashboards" / "demo.json"
    manifest_file = output_dir / "manifest.json"

    assert alerts_file.exists()
    assert dashboard_file.exists()
    assert manifest_file.exists()

    rendered = alerts_file.read_text(encoding="utf-8")
    assert "groups:" in rendered
    assert "DemoAlert" in rendered

    dashboard_payload = json.loads(dashboard_file.read_text(encoding="utf-8"))
    assert dashboard_payload["uid"] == "demo"
    assert manifest["dashboards"] == ["demo"]
    assert len(manifest["metrics"]) == 2
