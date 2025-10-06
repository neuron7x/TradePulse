# SPDX-License-Identifier: MIT
"""Multi-scale Kuramoto order parameter analysis utilities.

This module extends the classic Kuramoto order parameter by operating the
estimation at several temporal resolutions simultaneously.  Each timeframe is
analysed using the instantaneous phase of the price series and then aggregated
into a consensus synchronisation score.  An adaptive window length derived from
wavelet analysis keeps the method responsive to regime changes while remaining
robust to short lived noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd

from .base import BaseFeature, FeatureResult
from .kuramoto import compute_phase

try:  # SciPy is optional in lightweight environments.
    from scipy import signal as _signal  # type: ignore
except Exception:  # pragma: no cover - exercised in dedicated fallback test.
    _signal = None


class TimeFrame(Enum):
    """Supported timeframes expressed in seconds."""

    M1 = 60
    M5 = 300
    M15 = 900
    H1 = 3600

    @property
    def seconds(self) -> int:
        return int(self.value)


@dataclass(slots=True)
class KuramotoResult:
    """Container for the outcome of a single timeframe analysis."""

    R: float
    psi: float
    phases: np.ndarray
    timeframe: TimeFrame
    window_size: int


@dataclass(slots=True)
class MultiScaleResult:
    """Collective summary produced by :class:`MultiScaleKuramoto`."""

    consensus_R: float
    timeframe_results: Mapping[TimeFrame, KuramotoResult]
    dominant_scale: Optional[TimeFrame]
    cross_scale_coherence: float
    adaptive_window: int

    def as_dict(self) -> Dict[str, float | int | str | List[str]]:
        """Return a serialisable view of the result for downstream usage."""

        ordered = dict(sorted(self.timeframe_results.items(), key=lambda kv: kv[0].value))
        timeframe_names = [tf.name for tf in ordered]
        per_scale_r = {tf.name: res.R for tf, res in ordered.items()}
        payload: Dict[str, float | int | str | List[str]] = {
            "consensus_R": float(self.consensus_R),
            "cross_scale_coherence": float(self.cross_scale_coherence),
            "adaptive_window": int(self.adaptive_window),
            "timeframes": timeframe_names,
        }
        if self.dominant_scale is not None:
            payload["dominant_scale"] = self.dominant_scale.name
        payload.update({f"R_{name}": value for name, value in per_scale_r.items()})
        return payload


class WaveletWindowSelector:
    """Adaptive window selection using the Continuous Wavelet Transform."""

    def __init__(self, *, min_window: int = 50, max_window: int = 500) -> None:
        if min_window <= 0 or max_window <= 0:
            raise ValueError("Window bounds must be positive integers")
        if min_window >= max_window:
            raise ValueError("min_window must be strictly less than max_window")
        self.min_window = int(min_window)
        self.max_window = int(max_window)

    def select_window(self, prices: np.ndarray) -> int:
        """Return the dominant window length inferred from the price series."""

        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1:
            raise ValueError("prices must be a one-dimensional array")
        if prices.size < self.min_window:
            return self.min_window

        if _signal is None:
            # When SciPy is unavailable fall back to the geometric mean of the
            # bounds; this keeps the algorithm usable albeit without adaptivity.
            return int(np.sqrt(self.min_window * self.max_window))

        widths = np.arange(max(1, self.min_window // 4), max(2, self.max_window // 4))
        cwt_matrix = _signal.cwt(prices, _signal.morlet2, widths)
        power = np.abs(cwt_matrix) ** 2
        avg_power = power.mean(axis=1)
        dominant_scale_idx = int(np.argmax(avg_power))
        optimal_window = int(widths[dominant_scale_idx] * 2)
        return int(np.clip(optimal_window, self.min_window, self.max_window))


class MultiScaleKuramoto:
    """Kuramoto order parameter analysis executed on multiple timeframes."""

    def __init__(
        self,
        *,
        timeframes: Optional[Iterable[TimeFrame]] = None,
        use_adaptive_window: bool = True,
        base_window: int = 200,
        window_selector: Optional[WaveletWindowSelector] = None,
    ) -> None:
        self.timeframes: tuple[TimeFrame, ...] = tuple(timeframes or (
            TimeFrame.M1,
            TimeFrame.M5,
            TimeFrame.M15,
            TimeFrame.H1,
        ))
        if not self.timeframes:
            raise ValueError("At least one timeframe must be provided")
        if base_window <= 0:
            raise ValueError("base_window must be positive")
        self.use_adaptive_window = use_adaptive_window
        self.base_window = int(base_window)
        self.window_selector = window_selector or WaveletWindowSelector()

    def analyze(self, df: pd.DataFrame, *, price_col: str = "close") -> MultiScaleResult:
        """Execute the multi-scale analysis for the provided price series."""

        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in dataframe")
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        prices = df[price_col].astype(float)
        window = self._select_window(prices.values)

        timeframe_results: MutableMapping[TimeFrame, KuramotoResult] = {}
        for timeframe in self.timeframes:
            series = self._resample(prices, timeframe)
            if series.size < max(window, 8):
                continue
            timeframe_results[timeframe] = self._analyze_timeframe(series.values, timeframe, window)

        consensus = self._compute_consensus(timeframe_results)
        dominant_scale = self._dominant_scale(timeframe_results)
        coherence = self._cross_scale_coherence(timeframe_results)
        result = MultiScaleResult(
            consensus_R=consensus,
            timeframe_results=dict(timeframe_results),
            dominant_scale=dominant_scale,
            cross_scale_coherence=coherence,
            adaptive_window=window,
        )
        return result

    def _kuramoto_order_parameter(self, phases: np.ndarray) -> tuple[float, float]:
        """Return the Kuramoto order parameter ``R`` and mean phase ``ψ``."""

        values = np.asarray(phases, dtype=float)
        if values.size == 0:
            return 0.0, 0.0

        complex_order = np.mean(np.exp(1j * values))
        R = float(np.clip(np.abs(complex_order), 0.0, 1.0))
        psi = float(np.angle(complex_order))
        return R, psi

    def _select_window(self, prices: np.ndarray) -> int:
        if not self.use_adaptive_window:
            return self.base_window
        return int(self.window_selector.select_window(prices))

    def _resample(self, prices: pd.Series, timeframe: TimeFrame) -> pd.Series:
        rule = f"{timeframe.seconds}s"
        resampled = prices.resample(rule).last().dropna()
        if resampled.size == 0:
            return prices.iloc[[-1]]
        return resampled

    def _analyze_timeframe(self, prices: np.ndarray, timeframe: TimeFrame, window: int) -> KuramotoResult:
        phases = compute_phase(prices)
        if phases.size < window:
            window = max(1, phases.size)
        r_values: List[float] = []
        psi_values: List[float] = []
        for end in range(window, phases.size + 1):
            window_phases = phases[end - window : end]
            R, psi = self._kuramoto_order_parameter(window_phases)
            r_values.append(R)
            psi_values.append(psi)

        R_current = r_values[-1] if r_values else 0.0
        psi_current = psi_values[-1] if psi_values else 0.0
        return KuramotoResult(
            R=R_current,
            psi=psi_current,
            phases=phases,
            timeframe=timeframe,
            window_size=window,
        )

    def _compute_consensus(self, results: Mapping[TimeFrame, KuramotoResult]) -> float:
        if not results:
            return 0.0
        weights = {
            TimeFrame.M1: 0.1,
            TimeFrame.M5: 0.2,
            TimeFrame.M15: 0.3,
            TimeFrame.H1: 0.4,
        }
        weighted_sum = 0.0
        total_weight = 0.0
        for timeframe, result in results.items():
            weight = weights.get(timeframe, 1.0 / max(1, len(results)))
            weighted_sum += weight * result.R
            total_weight += weight
        return float(weighted_sum / total_weight) if total_weight else 0.0

    def _dominant_scale(self, results: Mapping[TimeFrame, KuramotoResult]) -> Optional[TimeFrame]:
        if not results:
            return None
        return max(results.items(), key=lambda item: item[1].R)[0]

    def _cross_scale_coherence(self, results: Mapping[TimeFrame, KuramotoResult]) -> float:
        if len(results) < 2:
            return 1.0 if results else 0.0
        values = np.array([res.R for res in results.values()], dtype=float)
        mean = float(values.mean())
        if mean == 0.0:
            return 0.0
        std = float(values.std())
        cv = std / mean
        return float(np.exp(-cv))


class MultiScaleKuramotoFeature(BaseFeature):
    """Feature wrapper exposing the consensus Kuramoto score."""

    def __init__(
        self,
        analyzer: Optional[MultiScaleKuramoto] = None,
        *,
        name: str | None = None,
        price_col: str = "close",
    ) -> None:
        super().__init__(name or "multiscale_kuramoto")
        self.analyzer = analyzer or MultiScaleKuramoto()
        self.price_col = price_col

    def transform(self, data: pd.DataFrame, **_: object) -> FeatureResult:
        result = self.analyzer.analyze(data, price_col=self.price_col)
        payload = result.as_dict()
        return FeatureResult(name=self.name, value=result.consensus_R, metadata=payload)


__all__ = [
    "KuramotoResult",
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
]
