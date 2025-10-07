"""Composite indicator blending Kuramoto and Ricci analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .multiscale_kuramoto import MultiScaleKuramoto, MultiScaleResult
from .temporal_ricci import TemporalRicciAnalyzer, TemporalRicciResult


class MarketPhase(Enum):
    CHAOTIC = "chaotic"
    PROTO_EMERGENT = "proto_emergent"
    STRONG_EMERGENT = "strong_emergent"
    TRANSITION = "transition"
    POST_EMERGENT = "post_emergent"


@dataclass(slots=True)
class CompositeSignal:
    phase: MarketPhase
    confidence: float
    kuramoto_R: float
    consensus_R: float
    cross_scale_coherence: float
    static_ricci: float
    temporal_ricci: float
    topological_transition: float
    entry_signal: float
    exit_signal: float
    risk_multiplier: float
    dominant_timeframe_sec: Optional[int]
    timestamp: pd.Timestamp
    skipped_timeframes: list[str] = field(default_factory=list)


class KuramotoRicciComposite:
    """Heuristic fusion of Kuramoto synchrony and Ricci curvature signals."""

    def __init__(
        self,
        R_strong_emergent: float = 0.8,
        R_proto_emergent: float = 0.4,
        coherence_threshold: float = 0.6,
        ricci_negative_threshold: float = -0.3,
        temporal_ricci_threshold: float = -0.2,
        transition_threshold: float = 0.7,
        min_confidence: float = 0.5,
    ) -> None:
        self.Rs = float(R_strong_emergent)
        self.Rp = float(R_proto_emergent)
        self.coh_min = float(coherence_threshold)
        self.kneg = float(ricci_negative_threshold)
        self.kt_thr = float(temporal_ricci_threshold)
        self.trans_thr = float(transition_threshold)
        self.min_conf = float(min_confidence)

    # ------------------------------------------------------------------
    # Internal decision helpers
    # ------------------------------------------------------------------
    def _phase(self, R: float, kt: float, transition: float, static_ricci: float) -> MarketPhase:
        if R > self.Rs and static_ricci < self.kneg and kt < self.kt_thr and transition < 0.5:
            return MarketPhase.STRONG_EMERGENT
        if transition > self.trans_thr:
            return MarketPhase.TRANSITION
        if self.Rp < R <= self.Rs and transition < 0.5 and static_ricci < 0:
            return MarketPhase.PROTO_EMERGENT
        if R > self.Rp and (static_ricci > 0 or kt > 0):
            return MarketPhase.POST_EMERGENT
        return MarketPhase.CHAOTIC

    def _confidence(self, phase: MarketPhase, coherence: float, transition: float, R: float) -> float:
        confidence = float(coherence)
        if phase is MarketPhase.STRONG_EMERGENT:
            confidence *= 1.0 + R
        elif phase is MarketPhase.CHAOTIC:
            confidence *= 0.5
        elif phase is MarketPhase.TRANSITION:
            confidence *= 0.5 + 0.5 * transition
        if coherence < self.coh_min:
            confidence *= max(0.0, coherence / max(self.coh_min, 1e-9))
        distance = min(abs(R - self.Rs), abs(R - self.Rp))
        if distance < 0.1:
            confidence *= 0.8
        return float(np.clip(confidence, 0.0, 1.0))

    def _entry(self, phase: MarketPhase, R: float, kt: float, confidence: float) -> float:
        if confidence < self.min_conf:
            return 0.0
        signal = 0.0
        if phase is MarketPhase.STRONG_EMERGENT:
            signal = np.clip(-kt, 0.0, 1.0)
        elif phase is MarketPhase.PROTO_EMERGENT:
            signal = 0.5 * R
        elif phase is MarketPhase.POST_EMERGENT:
            signal = -0.3
        return float(np.clip(signal * confidence, -1.0, 1.0))

    def _exit(self, phase: MarketPhase, transition: float, R: float) -> float:
        if phase is MarketPhase.POST_EMERGENT:
            return 0.7
        if phase is MarketPhase.TRANSITION:
            return float(np.clip(transition, 0.0, 1.0))
        if phase is MarketPhase.CHAOTIC:
            return 0.5
        if phase is MarketPhase.STRONG_EMERGENT:
            return 0.1
        return 0.3

    def _risk(self, phase: MarketPhase, confidence: float, coherence: float) -> float:
        base = 1.0
        if phase is MarketPhase.STRONG_EMERGENT:
            base = 1.0 + 0.5 * confidence
        elif phase is MarketPhase.PROTO_EMERGENT:
            base = 0.7 + 0.3 * confidence
        elif phase in (MarketPhase.TRANSITION, MarketPhase.CHAOTIC):
            base = 0.3
        elif phase is MarketPhase.POST_EMERGENT:
            base = 0.2
        return float(np.clip(base * coherence, 0.1, 2.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        kres: MultiScaleResult,
        rres: TemporalRicciResult,
        static_ricci: float,
        timestamp: pd.Timestamp,
    ) -> CompositeSignal:
        R = float(kres.consensus_R)
        coherence = float(kres.cross_scale_coherence)
        temporal_ricci = float(rres.temporal_curvature)
        transition = float(rres.topological_transition_score)
        phase = self._phase(R, temporal_ricci, transition, static_ricci)
        confidence = self._confidence(phase, coherence, transition, R)
        entry = self._entry(phase, R, temporal_ricci, confidence)
        exit_level = self._exit(phase, transition, R)
        risk = self._risk(phase, confidence, coherence)
        skipped = [tf.name for tf in getattr(kres, "skipped_timeframes", [])]
        return CompositeSignal(
            phase=phase,
            confidence=confidence,
            kuramoto_R=R,
            consensus_R=R,
            cross_scale_coherence=coherence,
            static_ricci=float(static_ricci),
            temporal_ricci=temporal_ricci,
            topological_transition=transition,
            entry_signal=entry,
            exit_signal=exit_level,
            risk_multiplier=risk,
            dominant_timeframe_sec=kres.dominant_scale_sec,
            timestamp=timestamp,
            skipped_timeframes=skipped,
        )

    def to_dict(self, signal: CompositeSignal) -> Dict[str, object]:
        return {
            "timestamp": signal.timestamp,
            "phase": signal.phase.value,
            "confidence": signal.confidence,
            "entry_signal": signal.entry_signal,
            "exit_signal": signal.exit_signal,
            "risk_multiplier": signal.risk_multiplier,
            "kuramoto_R": signal.kuramoto_R,
            "consensus_R": signal.consensus_R,
            "coherence": signal.cross_scale_coherence,
            "static_ricci": signal.static_ricci,
            "temporal_ricci": signal.temporal_ricci,
            "topological_transition": signal.topological_transition,
            "dominant_timeframe_sec": signal.dominant_timeframe_sec,
            "skipped_timeframes": signal.skipped_timeframes,
        }

    # Legacy compatibility hooks used by unit tests
    def _determine_phase(self, R: float, temporal_ricci: float, transition_score: float, static_ricci: float) -> MarketPhase:
        return self._phase(R, temporal_ricci, transition_score, static_ricci)

    def _compute_confidence(self, phase: MarketPhase, coherence: float, transition_score: float, R: float) -> float:
        return self._confidence(phase, coherence, transition_score, R)

    def _generate_entry_signal(
        self,
        phase: MarketPhase,
        R: float,
        temporal_ricci: float,
        transition_score: float,
        confidence: float,
    ) -> float:
        return self._entry(phase, R, temporal_ricci, confidence)

    def _generate_exit_signal(self, phase: MarketPhase, transition_score: float, R: float) -> float:
        return self._exit(phase, transition_score, R)

    def _compute_risk_multiplier(self, phase: MarketPhase, confidence: float, coherence: float) -> float:
        return self._risk(phase, confidence, coherence)


class TradePulseCompositeEngine:
    """High level helper orchestrating the full composite pipeline."""

    def __init__(
        self,
        kuramoto_config: Optional[Dict] = None,
        ricci_config: Optional[Dict] = None,
        composite_config: Optional[Dict] = None,
    ) -> None:
        self.k = MultiScaleKuramoto(**(kuramoto_config or {}))
        self.r = TemporalRicciAnalyzer(**(ricci_config or {}))
        self.c = KuramotoRicciComposite(**(composite_config or {}))
        self._history: list[CompositeSignal] = []

    def analyze_market(self, df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume") -> CompositeSignal:
        kres = self.k.analyze(df, price_col=price_col)
        rres = self.r.analyze(df, price_col=price_col, volume_col=volume_col)
        static_ricci = rres.graph_snapshots[-1].avg_curvature if rres.graph_snapshots else 0.0
        signal = self.c.analyze(kres, rres, static_ricci, df.index[-1])
        self._history.append(signal)
        return signal

    def get_signal_dataframe(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([self.c.to_dict(sig) for sig in self._history])

    @property
    def signal_history(self) -> list[CompositeSignal]:
        return list(self._history)
