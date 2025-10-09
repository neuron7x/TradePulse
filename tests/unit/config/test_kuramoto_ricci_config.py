# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from core.config import (
    ConfigError,
    KuramotoRicciIntegrationConfig,
    load_kuramoto_ricci_config,
)
from core.indicators.multiscale_kuramoto import TimeFrame


def test_loader_returns_defaults_when_file_missing() -> None:
    cfg = load_kuramoto_ricci_config("does-not-exist.yaml")
    engine_kwargs = cfg.to_engine_kwargs()

    assert engine_kwargs["kuramoto_config"]["base_window"] == 200
    assert engine_kwargs["ricci_config"]["window_size"] == 100
    assert engine_kwargs["composite_config"]["R_strong_emergent"] == 0.8


def test_loader_reads_yaml_values() -> None:
    payload = {
        "kuramoto": {
            "timeframes": ["M1", "M5"],
            "adaptive_window": {"enabled": False, "base_window": 128},
            "min_samples_per_scale": 32,
        },
        "ricci": {
            "temporal": {"window_size": 150, "n_snapshots": 5, "retain_history": False},
            "graph": {"n_levels": 12, "connection_threshold": 0.2},
        },
        "composite": {
            "thresholds": {"R_strong_emergent": 0.9, "R_proto_emergent": 0.5, "coherence_min": 0.7},
            "signals": {"min_confidence": 0.65},
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        path.write_text(yaml.safe_dump(payload))
        cfg = KuramotoRicciIntegrationConfig.from_file(path)

    engine_kwargs = cfg.to_engine_kwargs()
    kuramoto = engine_kwargs["kuramoto_config"]
    ricci = engine_kwargs["ricci_config"]
    composite = engine_kwargs["composite_config"]

    assert kuramoto["timeframes"] == (TimeFrame.M1, TimeFrame.M5)
    assert kuramoto["use_adaptive_window"] is False
    assert kuramoto["base_window"] == 128
    assert kuramoto["min_samples_per_scale"] == 32

    assert ricci["window_size"] == 150
    assert ricci["n_snapshots"] == 5
    assert ricci["retain_history"] is False
    assert ricci["connection_threshold"] == pytest.approx(0.2)

    assert composite["R_strong_emergent"] == 0.9
    assert composite["coherence_threshold"] == 0.7
    assert composite["min_confidence"] == pytest.approx(0.65)


def test_invalid_thresholds_raise_error() -> None:
    payload = {
        "composite": {"thresholds": {"R_proto_emergent": 0.8, "R_strong_emergent": 0.7}},
    }

    with pytest.raises(ConfigError):
        KuramotoRicciIntegrationConfig.from_mapping(payload)
