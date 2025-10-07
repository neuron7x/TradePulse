"""Multi-scale Kuramoto indicator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseFeature, FeatureResult
from .kuramoto import compute_phase

try:  # SciPy is optional; fall back to a deterministic heuristic when missing.
    from scipy import signal as _signal  # type: ignore
except Exception:  # pragma: no cover - exercised in dedicated fallback test.
    _signal = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TimeFrame(Enum):
    """Supported timeframes expressed in seconds."""

    M1 = 60
    M5 = 300
    M15 = 900
    M30 = 1800
    H1 = 3600
    H4 = 14_400
    D1 = 86_400

    @property
    def seconds(self) -> int:
        return int(self.value)

    @classmethod
    def coerce(cls, value: Union["TimeFrame", int]) -> "TimeFrame":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            for frame in cls:
                if frame.value == value:
                    return frame
        raise ValueError(f"Unsupported timeframe value: {value!r}")


@dataclass(slots=True)
class KuramotoResult:
    """Outcome of analysing a single timeframe."""

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
    skipped_timeframes: Sequence[TimeFrame]

    def as_dict(self) -> Dict[str, float | int | str | List[str]]:
        ordered = dict(sorted(self.timeframe_results.items(), key=lambda kv: kv[0].value))
        payload: Dict[str, float | int | str | List[str]] = {
            "consensus_R": float(self.consensus_R),
            "cross_scale_coherence": float(self.cross_scale_coherence),
            "adaptive_window": int(self.adaptive_window),
            "timeframes": [tf.name for tf in ordered],
        }
        if self.dominant_scale is not None:
            payload["dominant_scale"] = self.dominant_scale.name
        if self.skipped_timeframes:
            payload["skipped_timeframes"] = [tf.name for tf in self.skipped_timeframes]
        for tf, result in ordered.items():
            payload[f"R_{tf.name}"] = float(result.R)
            payload[f"psi_{tf.name}"] = float(result.psi)
        return payload

    @property
    def dominant_scale_sec(self) -> Optional[int]:
        return self.dominant_scale.seconds if self.dominant_scale is not None else None


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
class WaveletWindowSelector:
    """Adaptive window selection based on wavelet energy concentration."""

    def __init__(self, *, min_window: int = 50, max_window: int = 500) -> None:
        if min_window <= 0 or max_window <= 0:
            raise ValueError("Window bounds must be positive")
        if min_window >= max_window:
            raise ValueError("min_window must be strictly less than max_window")
        self.min_window = int(min_window)
        self.max_window = int(max_window)

    def select_window(self, prices: Sequence[float]) -> int:
        series = np.asarray(prices, dtype=float)
        if series.ndim != 1:
            raise ValueError("prices must be one-dimensional")
        if series.size < self.min_window:
            return self.min_window

        if _signal is None:
            return int(np.sqrt(self.min_window * self.max_window))

        widths = np.arange(max(1, self.min_window // 4), max(2, self.max_window // 4))
        cwt_matrix = _signal.cwt(series - series.mean(), _signal.morlet2, widths)
        power = np.abs(cwt_matrix) ** 2
        idx = int(np.argmax(power.mean(axis=1)))
        optimal = int(widths[idx] * 2)
        return int(np.clip(optimal, self.min_window, self.max_window))


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------
class MultiScaleKuramoto:
    """Kuramoto order parameter computed across multiple timeframes."""

    def __init__(
        self,
        timeframes: Optional[Iterable[Union[TimeFrame, int]]] = None,
        *,
        use_adaptive_window: bool = True,
        base_window: int = 200,
        window_selector: Optional[WaveletWindowSelector] = None,
    ) -> None:
        frames = tuple(TimeFrame.coerce(tf) for tf in (timeframes or (TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1)))
        if not frames:
            raise ValueError("At least one timeframe must be provided")
        if base_window <= 0:
            raise ValueError("base_window must be positive")
        self.timeframes: tuple[TimeFrame, ...] = frames
        self.use_adaptive_window = bool(use_adaptive_window)
        self.base_window = int(base_window)
        self.window_selector = window_selector or WaveletWindowSelector()

    def analyze(self, df: pd.DataFrame, *, price_col: str = "close") -> MultiScaleResult:
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in dataframe")
        if df.empty:
            raise ValueError("Dataframe cannot be empty")

        prices = df[price_col].astype(float)
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index)

        window = self._select_window(prices.values)

        timeframe_results: MutableMapping[TimeFrame, KuramotoResult] = {}
        for timeframe in self.timeframes:
            series = self._resample(prices, timeframe)
            if series.size < max(window, 8):
                continue
            timeframe_results[timeframe] = self._analyze_timeframe(series.values, timeframe, window)

        consensus = self._compute_consensus(timeframe_results)
        dominant = self._dominant_scale(timeframe_results)
        coherence = self._cross_scale_coherence(timeframe_results)
        skipped = [tf for tf in self.timeframes if tf not in timeframe_results]
        return MultiScaleResult(
            consensus_R=consensus,
            timeframe_results=dict(timeframe_results),
            dominant_scale=dominant,
            cross_scale_coherence=coherence,
            adaptive_window=window,
            skipped_timeframes=tuple(skipped),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_window(self, prices: np.ndarray) -> int:
        if not self.use_adaptive_window:
            return self.base_window
        return int(self.window_selector.select_window(prices))

    def _resample(self, prices: pd.Series, timeframe: TimeFrame) -> pd.Series:
        rule = f"{timeframe.seconds}s"
        resampled = prices.resample(rule).last().dropna()
        if resampled.empty:
            return prices.iloc[[-1]]
        return resampled

    def _analyze_timeframe(self, prices: np.ndarray, timeframe: TimeFrame, window: int) -> KuramotoResult:
        phases = compute_phase(prices)
        if phases.size < window:
            window = max(1, phases.size)
        r_values: List[float] = []
        psi_values: List[float] = []
        for end in range(window, phases.size + 1):
            segment = phases[end - window : end]
            R, psi = self._kuramoto_order_parameter(segment)
            r_values.append(R)
            psi_values.append(psi)
        R_current = r_values[-1] if r_values else 0.0
        psi_current = psi_values[-1] if psi_values else 0.0
        return KuramotoResult(R=R_current, psi=psi_current, phases=phases, timeframe=timeframe, window_size=window)

    @staticmethod
    def _kuramoto_order_parameter(phases: np.ndarray) -> Tuple[float, float]:
        values = np.asarray(phases, dtype=float)
        if values.size == 0:
            return 0.0, 0.0
        complex_order = np.mean(np.exp(1j * values))
        R = float(np.clip(np.abs(complex_order), 0.0, 1.0))
        psi = float(np.angle(complex_order))
        return R, psi

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


# ---------------------------------------------------------------------------
# Feature adapter
# ---------------------------------------------------------------------------
class MultiScaleKuramotoFeature(BaseFeature):
    """Expose the multi-scale Kuramoto consensus as a feature."""

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
        metadata = result.as_dict()
        return FeatureResult(name=self.name, value=result.consensus_R, metadata=metadata)


__all__ = [
    "KuramotoResult",
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
]
