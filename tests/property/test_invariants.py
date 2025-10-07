"""Basic invariants for the Kuramoto–Ricci composite configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path("configs/kuramoto_ricci_composite.yaml")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_kuramoto_timeframes_are_increasing() -> None:
    config = load_config()
    frames = config["kuramoto"]["timeframes"]
    assert all(isinstance(value, int) and value > 0 for value in frames)
    assert frames == sorted(frames)


def test_adaptive_window_bounds() -> None:
    config = load_config()
    adaptive = config["kuramoto"].get("adaptive_window", {})
    assert adaptive.get("enabled", True) is True
    assert adaptive.get("min_window", 0) < adaptive.get("max_window", 0)
    assert adaptive.get("base_window", 0) >= adaptive.get("min_window", 0)


def test_temporal_ricci_parameters_valid() -> None:
    config = load_config()
    temporal = config["ricci"]["temporal"]
    assert temporal["window_size"] > 0
    assert 2 <= temporal["n_snapshots"] <= 32


def test_composite_thresholds_are_ordered() -> None:
    config = load_config()
    thresholds = config["composite"]["thresholds"]
    assert 0.0 < thresholds["R_proto_emergent"] < thresholds["R_strong_emergent"] <= 1.0
    assert thresholds["coherence_min"] <= thresholds["R_strong_emergent"]
    assert thresholds["ricci_negative"] < 0 < thresholds["topological_transition"] <= 1.0
    assert thresholds["temporal_ricci"] < 0


def test_signal_confidence_is_probabilistic() -> None:
    config = load_config()
    min_confidence = config["composite"]["signals"]["min_confidence"]
    assert 0.0 <= min_confidence <= 1.0
