# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators.kuramoto_ricci_composite import (
    CompositeSignal,
    KuramotoRicciComposite,
    MarketPhase,
    TradePulseCompositeEngine,
)
from core.indicators.multiscale_kuramoto import KuramotoResult, MultiScaleResult, TimeFrame
from core.indicators.temporal_ricci import TemporalRicciResult


def _multi_scale_result(R: float, coherence: float, timeframe: TimeFrame) -> MultiScaleResult:
    phases = np.zeros(8)
    result = KuramotoResult(R=R, psi=0.0, phases=phases, timeframe=timeframe, window_size=16)
    timeframe_results = {timeframe: result}
    return MultiScaleResult(
        consensus_R=R,
        timeframe_results=timeframe_results,
        dominant_scale=timeframe,
        cross_scale_coherence=coherence,
        adaptive_window=32,
    )


def _temporal_result(temporal_curvature: float, transition_score: float) -> TemporalRicciResult:
    return TemporalRicciResult(
        temporal_curvature=temporal_curvature,
        topological_transition_score=transition_score,
        graph_snapshots=[],
        structural_stability=0.8,
        edge_persistence=0.6,
    )


def test_composite_identifies_strong_emergent_phase() -> None:
    composite = KuramotoRicciComposite()
    multi = _multi_scale_result(R=0.88, coherence=0.85, timeframe=TimeFrame.M1)
    temporal = _temporal_result(temporal_curvature=-0.35, transition_score=0.2)

    signal = composite.analyze(multi, temporal, static_ricci=-0.45, timestamp=pd.Timestamp("2024-01-01"))

    assert signal.phase is MarketPhase.STRONG_EMERGENT
    assert signal.entry_signal > 0.0
    assert signal.dominant_timeframe == TimeFrame.M1.name
    assert pytest.approx(signal.kuramoto_R, rel=1e-6) == 0.88


def test_composite_marks_transition_on_high_topological_shift() -> None:
    composite = KuramotoRicciComposite()
    multi = _multi_scale_result(R=0.65, coherence=0.6, timeframe=TimeFrame.M5)
    temporal = _temporal_result(temporal_curvature=-0.1, transition_score=0.9)

    signal = composite.analyze(multi, temporal, static_ricci=-0.2, timestamp=pd.Timestamp("2024-01-02"))

    assert signal.phase is MarketPhase.TRANSITION
    assert signal.exit_signal >= 0.8
    assert signal.confidence <= 1.0


def test_trade_pulse_engine_generates_history() -> None:
    pytest.importorskip("networkx")

    rng = np.random.default_rng(123)
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.2, 512))
    volumes = rng.lognormal(mean=1.0, sigma=0.25, size=512)
    index = pd.date_range("2024-01-01", periods=512, freq="1min")
    df = pd.DataFrame({"close": prices, "volume": volumes}, index=index)

    engine = TradePulseCompositeEngine(
        kuramoto_config={
            "timeframes": (TimeFrame.M1, TimeFrame.M5),
            "use_adaptive_window": False,
            "base_window": 64,
        },
        ricci_config={
            "window_size": 64,
            "n_snapshots": 4,
            "n_levels": 10,
            "connection_threshold": 0.05,
        },
        composite_config={"min_confidence": 0.0},
    )

    signal = engine.analyze_market(df)

    assert isinstance(signal, CompositeSignal)
    assert engine.signal_history[-1] is signal

    frame = engine.get_signal_dataframe()
    assert not frame.empty
    assert "phase" in frame.columns
    assert frame.iloc[-1]["phase"] == signal.phase.value

    engine.clear_history()
    assert engine.signal_history == []
