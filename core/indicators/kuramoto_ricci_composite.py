# SPDX-License-Identifier: MIT
"""Composite Kuramoto-Ricci market regime indicator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .multiscale_kuramoto import MultiScaleKuramoto, MultiScaleResult
from .temporal_ricci import TemporalRicciAnalyzer, TemporalRicciResult


class MarketPhase(Enum):
    """Enhanced market phases driven by synchronisation and topology."""

    CHAOTIC = "chaotic"
    PROTO_EMERGENT = "proto_emergent"
    STRONG_EMERGENT = "strong_emergent"
    TRANSITION = "transition"
    POST_EMERGENT = "post_emergent"


@dataclass(slots=True)
class CompositeSignal:
    """Complete composite signal output."""

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
    dominant_timeframe: str
    timestamp: pd.Timestamp


class KuramotoRicciComposite:
    """Combine Kuramoto synchronisation and Ricci curvature signals."""

    def __init__(
        self,
        *,
        R_strong_emergent: float = 0.8,
        R_proto_emergent: float = 0.4,
        coherence_threshold: float = 0.6,
        ricci_negative_threshold: float = -0.3,
        temporal_ricci_threshold: float = -0.2,
        transition_threshold: float = 0.7,
        min_confidence: float = 0.5,
    ) -> None:
        self.R_strong = float(R_strong_emergent)
        self.R_proto = float(R_proto_emergent)
        self.coherence_threshold = float(coherence_threshold)
        self.ricci_neg = float(ricci_negative_threshold)
        self.temporal_ricci_thresh = float(temporal_ricci_threshold)
        self.transition_thresh = float(transition_threshold)
        self.min_confidence = float(min_confidence)

    def _determine_phase(
        self,
        R: float,
        temporal_ricci: float,
        transition_score: float,
        static_ricci: float,
    ) -> MarketPhase:
        if (
            R > self.R_strong
            and static_ricci < self.ricci_neg
            and temporal_ricci < self.temporal_ricci_thresh
            and transition_score < 0.5
        ):
            return MarketPhase.STRONG_EMERGENT

        if transition_score > self.transition_thresh:
            return MarketPhase.TRANSITION

        if (
            self.R_proto < R <= self.R_strong
            and transition_score < 0.5
            and static_ricci < 0.0
        ):
            return MarketPhase.PROTO_EMERGENT

        if R > self.R_proto and (static_ricci > 0.0 or temporal_ricci > 0.0):
            return MarketPhase.POST_EMERGENT

        return MarketPhase.CHAOTIC

    def _compute_confidence(
        self,
        phase: MarketPhase,
        coherence: float,
        transition_score: float,
        R: float,
    ) -> float:
        coherence = float(np.clip(coherence, 0.0, 1.0))
        confidence = coherence

        if coherence < self.coherence_threshold:
            scale = coherence / max(self.coherence_threshold, 1e-6)
            confidence *= np.clip(scale, 0.0, 1.0)

        if phase == MarketPhase.STRONG_EMERGENT:
            confidence *= 1.0 + max(R, 0.0)
        elif phase == MarketPhase.CHAOTIC:
            confidence *= 0.5
        elif phase == MarketPhase.TRANSITION:
            confidence *= 0.5 + 0.5 * np.clip(transition_score, 0.0, 1.0)

        R_distance = min(abs(R - self.R_strong), abs(R - self.R_proto))
        if R_distance < 0.1:
            confidence *= 0.8

        return float(np.clip(confidence, 0.0, 1.0))

    def _generate_entry_signal(
        self,
        phase: MarketPhase,
        R: float,
        temporal_ricci: float,
        confidence: float,
    ) -> float:
        if confidence < self.min_confidence:
            return 0.0

        signal = 0.0
        if phase == MarketPhase.STRONG_EMERGENT:
            signal = np.clip(-temporal_ricci, 0.0, 1.0)
        elif phase == MarketPhase.PROTO_EMERGENT:
            signal = max(R, 0.0) * 0.5
        elif phase == MarketPhase.POST_EMERGENT:
            signal = -0.3

        signal *= confidence
        return float(np.clip(signal, -1.0, 1.0))

    def _generate_exit_signal(
        self,
        phase: MarketPhase,
        transition_score: float,
        R: float,
    ) -> float:
        exit_urgency = 0.0

        if phase == MarketPhase.POST_EMERGENT:
            exit_urgency = 0.7
        elif phase == MarketPhase.TRANSITION:
            exit_urgency = np.clip(transition_score, 0.0, 1.0)
        elif phase == MarketPhase.CHAOTIC:
            exit_urgency = 0.5
        elif phase == MarketPhase.STRONG_EMERGENT:
            exit_urgency = 0.1

        if R < self.R_proto:
            exit_urgency = max(exit_urgency, 0.5)

        return float(np.clip(exit_urgency, 0.0, 1.0))

    def _compute_risk_multiplier(
        self,
        phase: MarketPhase,
        confidence: float,
        coherence: float,
    ) -> float:
        base = 1.0
        if phase == MarketPhase.STRONG_EMERGENT:
            base = 1.0 + 0.5 * confidence
        elif phase == MarketPhase.PROTO_EMERGENT:
            base = 0.7 + 0.3 * confidence
        elif phase in (MarketPhase.TRANSITION, MarketPhase.CHAOTIC):
            base = 0.3
        elif phase == MarketPhase.POST_EMERGENT:
            base = 0.2

        multiplier = base * np.clip(coherence, 0.0, 1.0)
        return float(np.clip(multiplier, 0.1, 2.0))

    def analyze(
        self,
        kuramoto_result: MultiScaleResult,
        ricci_result: TemporalRicciResult,
        static_ricci: float,
        timestamp: pd.Timestamp,
    ) -> CompositeSignal:
        R = float(kuramoto_result.consensus_R)
        coherence = float(kuramoto_result.cross_scale_coherence)
        temporal_ricci = float(ricci_result.temporal_curvature)
        transition_score = float(ricci_result.topological_transition_score)

        phase = self._determine_phase(R, temporal_ricci, transition_score, float(static_ricci))
        confidence = self._compute_confidence(phase, coherence, transition_score, R)
        entry_signal = self._generate_entry_signal(phase, R, temporal_ricci, confidence)
        exit_signal = self._generate_exit_signal(phase, transition_score, R)
        risk_multiplier = self._compute_risk_multiplier(phase, confidence, coherence)

        dominant = kuramoto_result.dominant_scale
        dominant_name = dominant.name if dominant is not None else "UNDEFINED"
        dominant_r = (
            kuramoto_result.timeframe_results[dominant].R
            if dominant is not None and dominant in kuramoto_result.timeframe_results
            else R
        )

        return CompositeSignal(
            phase=phase,
            confidence=confidence,
            kuramoto_R=float(dominant_r),
            consensus_R=R,
            cross_scale_coherence=coherence,
            static_ricci=float(static_ricci),
            temporal_ricci=temporal_ricci,
            topological_transition=transition_score,
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            risk_multiplier=risk_multiplier,
            dominant_timeframe=dominant_name,
            timestamp=pd.Timestamp(timestamp),
        )

    def to_dict(self, signal: CompositeSignal) -> Dict[str, object]:
        return {
            "timestamp": pd.Timestamp(signal.timestamp),
            "phase": signal.phase.value,
            "confidence": float(signal.confidence),
            "entry_signal": float(signal.entry_signal),
            "exit_signal": float(signal.exit_signal),
            "risk_multiplier": float(signal.risk_multiplier),
            "kuramoto_R": float(signal.kuramoto_R),
            "consensus_R": float(signal.consensus_R),
            "coherence": float(signal.cross_scale_coherence),
            "static_ricci": float(signal.static_ricci),
            "temporal_ricci": float(signal.temporal_ricci),
            "topological_transition": float(signal.topological_transition),
            "dominant_timeframe": signal.dominant_timeframe,
        }


class TradePulseCompositeEngine:
    """High-level engine orchestrating the composite indicator workflow."""

    def __init__(
        self,
        *,
        kuramoto_config: Optional[Dict[str, object]] = None,
        ricci_config: Optional[Dict[str, object]] = None,
        composite_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.kuramoto_analyzer = MultiScaleKuramoto(**(kuramoto_config or {}))
        self.ricci_analyzer = TemporalRicciAnalyzer(**(ricci_config or {}))
        self.composite = KuramotoRicciComposite(**(composite_config or {}))
        self.signal_history: List[CompositeSignal] = []

    def analyze_market(
        self,
        df: pd.DataFrame,
        *,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
    ) -> CompositeSignal:
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in dataframe")

        volume_column: Optional[str]
        if volume_col is not None and volume_col in df.columns:
            volume_column = volume_col
        else:
            volume_column = None

        kuramoto_result = self.kuramoto_analyzer.analyze(df, price_col=price_col)
        ricci_result = self.ricci_analyzer.analyze(
            df,
            price_col=price_col,
            volume_col=volume_column,
        )

        static_ricci = (
            ricci_result.graph_snapshots[-1].avg_curvature
            if ricci_result.graph_snapshots
            else 0.0
        )

        signal = self.composite.analyze(
            kuramoto_result,
            ricci_result,
            static_ricci,
            pd.Timestamp(df.index[-1]),
        )
        self.signal_history.append(signal)
        return signal

    def get_signal_dataframe(self) -> pd.DataFrame:
        if not self.signal_history:
            return pd.DataFrame()
        records = [self.composite.to_dict(sig) for sig in self.signal_history]
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
        return frame

    def clear_history(self) -> None:
        self.signal_history.clear()


__all__ = [
    "MarketPhase",
    "CompositeSignal",
    "KuramotoRicciComposite",
    "TradePulseCompositeEngine",
]
