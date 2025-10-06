# SPDX-License-Identifier: MIT
"""Composite indicator combining Kuramoto synchronisation and Ricci curvature."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import List, Optional

import numpy as np
import pandas as pd

from .multiscale_kuramoto import MultiScaleKuramoto, MultiScaleResult
from .temporal_ricci import TemporalRicciAnalyzer, TemporalRicciResult
from .ricci import build_price_graph, mean_ricci


class MarketPhase(Enum):
    """High level market regimes detected by the composite engine."""

    CHAOTIC = auto()
    TRANSITION = auto()
    STRONG_EMERGENT = auto()
    POST_EMERGENT = auto()
    STABLE = auto()


@dataclass(slots=True)
class CompositeSignal:
    """Signal emitted by :class:`TradePulseCompositeEngine`."""

    phase: MarketPhase
    confidence: float
    entry_signal: float
    exit_signal: float
    risk_multiplier: float
    kuramoto_R: float
    temporal_ricci: float
    transition_score: float
    static_ricci: float
    timestamp: Optional[pd.Timestamp] = None

    def as_dict(self) -> dict[str, float | str | None]:
        payload = asdict(self)
        payload["phase"] = self.phase.name
        if self.timestamp is not None:
            payload["timestamp"] = pd.Timestamp(self.timestamp)
        return payload


class KuramotoRicciComposite:
    """Logic that maps raw indicators into market phases and trading signals."""

    def __init__(
        self,
        *,
        R_strong_emergent: float = 0.8,
        R_emergent: float = 0.65,
        R_low: float = 0.35,
        ricci_negative_threshold: float = -0.25,
        transition_threshold: float = 0.65,
        post_transition_threshold: float = 0.45,
        min_confidence: float = 0.35,
    ) -> None:
        self.R_strong_emergent = float(R_strong_emergent)
        self.R_emergent = float(R_emergent)
        self.R_low = float(R_low)
        self.ricci_negative_threshold = float(ricci_negative_threshold)
        self.transition_threshold = float(transition_threshold)
        self.post_transition_threshold = float(post_transition_threshold)
        self.min_confidence = float(min_confidence)

    def _determine_phase(
        self,
        *,
        R: float,
        temporal_ricci: float,
        transition_score: float,
        static_ricci: float,
    ) -> MarketPhase:
        transition_score = float(np.clip(transition_score, 0.0, 1.0))
        R = float(np.clip(R, 0.0, 1.0))

        if transition_score >= self.transition_threshold:
            return MarketPhase.TRANSITION

        min_ricci = min(float(temporal_ricci), float(static_ricci))
        if R >= self.R_strong_emergent and min_ricci <= self.ricci_negative_threshold:
            return MarketPhase.STRONG_EMERGENT

        if transition_score >= self.post_transition_threshold:
            return MarketPhase.POST_EMERGENT

        if R >= self.R_emergent:
            return MarketPhase.STABLE

        if R <= self.R_low:
            return MarketPhase.CHAOTIC

        return MarketPhase.STABLE

    def _compute_confidence(
        self,
        *,
        phase: MarketPhase,
        coherence: float,
        transition_score: float,
        R: float,
    ) -> float:
        coherence = float(np.clip(coherence, 0.0, 1.0))
        transition_score = float(np.clip(transition_score, 0.0, 1.0))
        R = float(np.clip(R, 0.0, 1.0))

        base = 0.5 * coherence + 0.3 * R + 0.2 * (1.0 - transition_score)
        if phase is MarketPhase.TRANSITION:
            base *= 0.75
        elif phase is MarketPhase.STRONG_EMERGENT:
            base = min(1.0, base + 0.1)
        return float(np.clip(base, 0.0, 1.0))

    def _generate_entry_signal(
        self,
        *,
        phase: MarketPhase,
        R: float,
        temporal_ricci: float,
        transition_score: float,
        confidence: float,
    ) -> float:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if confidence < self.min_confidence:
            return 0.0

        R = float(np.clip(R, 0.0, 1.0))
        transition_score = float(np.clip(transition_score, 0.0, 1.0))

        if phase is MarketPhase.STRONG_EMERGENT:
            curvature_boost = float(np.clip(-temporal_ricci, 0.0, 1.0))
            signal = confidence * (0.4 + 0.4 * R + 0.4 * curvature_boost)
        elif phase is MarketPhase.TRANSITION:
            signal = confidence * (0.5 - transition_score)
        elif phase is MarketPhase.CHAOTIC:
            signal = -confidence * (0.3 + 0.4 * (1.0 - R))
        else:
            signal = confidence * 0.5 * R

        return float(np.clip(signal, -1.0, 1.0))

    def _generate_exit_signal(
        self,
        *,
        phase: MarketPhase,
        transition_score: float,
        R: float,
    ) -> float:
        transition_score = float(np.clip(transition_score, 0.0, 1.0))
        R = float(np.clip(R, 0.0, 1.0))

        base = 0.6 * transition_score + 0.4 * (1.0 - R)
        if phase is MarketPhase.POST_EMERGENT:
            base = min(1.0, base + 0.2)
        elif phase is MarketPhase.STRONG_EMERGENT:
            base *= 0.5
        return float(np.clip(base, 0.0, 1.0))

    def _compute_risk_multiplier(
        self,
        *,
        phase: MarketPhase,
        confidence: float,
        coherence: float,
    ) -> float:
        confidence = float(np.clip(confidence, 0.0, 1.0))
        coherence = float(np.clip(coherence, 0.0, 1.0))

        base = 1.0 + 0.6 * (confidence - 0.5) + 0.4 * (coherence - 0.5)
        if phase is MarketPhase.STRONG_EMERGENT:
            base += 0.2 * confidence
        elif phase is MarketPhase.TRANSITION:
            base -= 0.3 * (1.0 - confidence)
        elif phase is MarketPhase.CHAOTIC:
            base -= 0.2
        return float(np.clip(base, 0.1, 2.0))


class TradePulseCompositeEngine:
    """High level orchestration of the composite indicator pipeline."""

    def __init__(
        self,
        *,
        kuramoto: Optional[MultiScaleKuramoto] = None,
        temporal_ricci: Optional[TemporalRicciAnalyzer] = None,
        composite: Optional[KuramotoRicciComposite] = None,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
    ) -> None:
        self.kuramoto = kuramoto or MultiScaleKuramoto()
        self.temporal_ricci = temporal_ricci or TemporalRicciAnalyzer()
        self.composite = composite or KuramotoRicciComposite()
        self.price_col = price_col
        self.volume_col = volume_col
        self.signal_history: List[CompositeSignal] = []

    def _static_ricci(self, prices: np.ndarray) -> float:
        graph = build_price_graph(prices)
        return float(mean_ricci(graph))

    def analyze_market(self, df: pd.DataFrame) -> CompositeSignal:
        if self.price_col not in df.columns:
            raise KeyError(f"Column '{self.price_col}' not found in dataframe")

        prices = df[self.price_col].astype(float)

        kuramoto_result: MultiScaleResult = self.kuramoto.analyze(df, price_col=self.price_col)
        temporal_result: TemporalRicciResult = self.temporal_ricci.analyze(
            df,
            price_col=self.price_col,
            volume_col=self.volume_col,
        )
        static_ricci = self._static_ricci(prices.to_numpy())

        phase = self.composite._determine_phase(
            R=kuramoto_result.consensus_R,
            temporal_ricci=temporal_result.temporal_curvature,
            transition_score=temporal_result.topological_transition_score,
            static_ricci=static_ricci,
        )

        confidence = self.composite._compute_confidence(
            phase=phase,
            coherence=kuramoto_result.cross_scale_coherence,
            transition_score=temporal_result.topological_transition_score,
            R=kuramoto_result.consensus_R,
        )

        entry_signal = self.composite._generate_entry_signal(
            phase=phase,
            R=kuramoto_result.consensus_R,
            temporal_ricci=temporal_result.temporal_curvature,
            transition_score=temporal_result.topological_transition_score,
            confidence=confidence,
        )

        exit_signal = self.composite._generate_exit_signal(
            phase=phase,
            transition_score=temporal_result.topological_transition_score,
            R=kuramoto_result.consensus_R,
        )

        risk_multiplier = self.composite._compute_risk_multiplier(
            phase=phase,
            confidence=confidence,
            coherence=kuramoto_result.cross_scale_coherence,
        )

        timestamp = prices.index[-1] if isinstance(prices.index, pd.DatetimeIndex) else None

        signal = CompositeSignal(
            phase=phase,
            confidence=confidence,
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            risk_multiplier=risk_multiplier,
            kuramoto_R=kuramoto_result.consensus_R,
            temporal_ricci=temporal_result.temporal_curvature,
            transition_score=temporal_result.topological_transition_score,
            static_ricci=static_ricci,
            timestamp=timestamp,
        )

        self.signal_history.append(signal)
        return signal

    def get_signal_dataframe(self) -> pd.DataFrame:
        if not self.signal_history:
            return pd.DataFrame(columns=[
                "phase",
                "confidence",
                "entry_signal",
                "exit_signal",
                "risk_multiplier",
                "kuramoto_R",
                "temporal_ricci",
                "transition_score",
                "static_ricci",
                "timestamp",
            ])
        records = [signal.as_dict() for signal in self.signal_history]
        df = pd.DataFrame.from_records(records)
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


__all__ = [
    "CompositeSignal",
    "KuramotoRicciComposite",
    "MarketPhase",
    "TradePulseCompositeEngine",
]
