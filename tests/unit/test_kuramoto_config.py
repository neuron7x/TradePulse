# SPDX-License-Identifier: MIT
"""Sanity checks for the Kuramoto-Ricci composite configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "kuramoto_ricci_composite.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_top_level_structure():
    config = _load_config()
    expected_keys = {
        "kuramoto",
        "ricci",
        "composite",
        "integration",
        "backtest",
        "monitoring",
        "features",
        "experimental",
        "metadata",
    }
    assert set(config) == expected_keys


def test_consensus_weights_sum_to_one():
    config = _load_config()
    weights = config["kuramoto"]["consensus_weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_thresholds_are_ordered():
    config = _load_config()
    thresholds = config["composite"]["thresholds"]
    assert thresholds["R_strong_emergent"] > thresholds["R_proto_emergent"]
    assert thresholds["coherence_min"] >= thresholds["R_proto_emergent"]
    assert thresholds["topological_transition"] > 0.5


def test_risk_multipliers_within_bounds():
    config = _load_config()
    multipliers = config["composite"]["signals"]["risk_multipliers"]
    min_cap = multipliers["min"]
    max_cap = multipliers["max"]
    assert 0 < min_cap < 1
    assert max_cap >= 2.0 - 1e-9
    for phase, payload in multipliers.items():
        if isinstance(payload, dict):
            assert min_cap <= payload["base"] <= max_cap
            if "confidence_scale" in payload:
                assert payload["confidence_scale"] >= 0


def test_monitoring_metrics_list():
    config = _load_config()
    metrics = config["monitoring"]["dashboard"]["metrics_to_display"]
    assert metrics[0] == "kuramoto_R"
    assert "topological_transition" in metrics
    assert len(metrics) == len(set(metrics))


def test_metadata_fields_present():
    config = _load_config()
    metadata = config["metadata"]
    for field in ("version", "author", "description", "created", "last_modified"):
        assert metadata[field]

