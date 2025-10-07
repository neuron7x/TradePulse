"""Multi-scale Kuramoto indicator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from .base import BaseFeature, FeatureResult

try:  # pragma: no cover - optional dependency
    from scipy import signal as _signal
except Exception:  # pragma: no cover - SciPy not available in minimal envs
    _signal = None


# -------------------- Lightweight Hilbert (no SciPy) --------------------
def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Compute analytic signal via FFT-based Hilbert transform."""
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.fft(x, n=n)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    xa = np.fft.ifft(X * h)
    return xa


def extract_phase(x: np.ndarray) -> np.ndarray:
    """Detrend (simple linear) + analytic signal -> angle."""
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n)
    # simple linear detrend
    A = np.vstack([t, np.ones(n)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    detrended = x - (m * t + b)
    z = analytic_signal(detrended)
    return np.angle(z)


# -------------------- Adaptive window via autocorrelation --------------------
def dominant_period_autocorr(x: np.ndarray, min_window: int = 50, max_window: int = 500) -> int:
    """Estimate dominant period using (biased) autocorrelation peak."""
    x = np.asarray(x, dtype=float)
    if np.allclose(x, x[0]):
        return min_window
    x = x - np.mean(x)
    n = x.size
    if n < min_window * 2:
        return max(min_window, min(n // 4, max_window))

    # FFT-based autocorr
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conjugate(f)).real[:n]
    ac = ac / ac[0] if ac[0] != 0 else ac

    # find first local maximum after lag 1
    start = max(2, min_window // 4)
    end = min(n // 2, max_window * 2)
    if end <= start + 1:
        return min_window
    lag = start + np.argmax(ac[start:end])
    win = int(np.clip(lag * 2, min_window, max_window))
    return win


# -------------------- Timeframe and results models --------------------
class TimeFrame(Enum):
    """Canonical exchange-style timeframes expressed in seconds."""

    M1 = 60
    M5 = 300
    M15 = 900
    M30 = 1800
    H1 = 3600
    H4 = 14400
    D1 = 86400

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @property
    def seconds(self) -> int:
        return int(self.value)

    @classmethod
    def from_any(cls, value: Union["TimeFrame", int]) -> "TimeFrame":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            try:
                return TimeFrame(value)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unsupported timeframe seconds: {value}") from exc
        raise TypeError(f"Unsupported timeframe type: {type(value)!r}")


@dataclass(slots=True)
class KuramotoResult:
    R: float
    psi: float
    phases: np.ndarray
    timeframe: TimeFrame
    window_size: int


@dataclass(slots=True)
class MultiScaleResult:
    consensus_R: float
    timeframe_results: Dict[TimeFrame, KuramotoResult]
    dominant_scale: Optional[TimeFrame]
    cross_scale_coherence: float
    adaptive_window: int
    skipped_timeframes: List[TimeFrame]


# -------------------- Window selection --------------------
class WaveletWindowSelector:
    """Select analysis window using wavelet energy concentration."""

    def __init__(self, min_window: int = 64, max_window: int = 512) -> None:
        if min_window <= 0 or max_window <= 0:
            raise ValueError("Window bounds must be positive")
        if min_window > max_window:
            raise ValueError("min_window cannot exceed max_window")
        self.min_window = int(min_window)
        self.max_window = int(max_window)

    def select_window(self, prices: Sequence[float]) -> int:
        series = np.asarray(prices, dtype=float)
        if series.size < self.min_window:
            return self.min_window
        if _signal is None:
            warnings.warn(
                "WaveletWindowSelector falling back to geometric mean window because SciPy is unavailable",
                RuntimeWarning,
                stacklevel=2,
            )
            return int(np.sqrt(self.min_window * self.max_window))

        detrended = series - np.mean(series)
        widths = np.linspace(self.min_window, self.max_window, num=32)
        coef = _signal.cwt(detrended, _signal.ricker, widths)
        power = np.abs(coef) ** 2
        idx = int(np.argmax(power.mean(axis=1)))
        window = int(widths[idx])
        return int(np.clip(window, self.min_window, self.max_window))


# -------------------- Core class --------------------
class MultiScaleKuramoto:
    """Multi-scale Kuramoto analysis with optional adaptive windowing."""

    def __init__(
        self,
        timeframes: Optional[Sequence[Union[TimeFrame, int]]] = None,
        use_adaptive_window: bool = True,
        base_window: int = 200,
        window_selector: Optional[WaveletWindowSelector] = None,
    ) -> None:
        selected = timeframes or (TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1)
        self.timeframes: Tuple[TimeFrame, ...] = tuple(TimeFrame.from_any(tf) for tf in selected)
        self.use_adaptive_window = use_adaptive_window
        self.base_window = int(base_window)
        self._window_selector = window_selector or WaveletWindowSelector(
            min_window=max(16, self.base_window // 2),
            max_window=max(self.base_window, self.base_window * 2),
        )

    @staticmethod
    def _kuramoto(phases: np.ndarray) -> Tuple[float, float]:
        z = np.exp(1j * phases).mean()
        return float(np.abs(z)), float(np.angle(z))

    def _kuramoto_order_parameter(self, phases: np.ndarray) -> Tuple[float, float]:
        phases = np.asarray(phases, dtype=float)
        if phases.ndim != 1:
            raise ValueError("phases must be one-dimensional")
        return self._kuramoto(phases)

    def _resample_close(self, df: pd.DataFrame, timeframe: TimeFrame, price_col: str) -> pd.Series:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df[price_col].resample(f"{timeframe.seconds}s").last().dropna()

    def analyze_single_timeframe(self, prices: np.ndarray, timeframe: TimeFrame, window: int) -> KuramotoResult:
        phases = extract_phase(prices)
        R_vals: List[float] = []
        psi_vals: List[float] = []
        for i in range(window, len(phases)):
            R, psi = self._kuramoto(phases[i - window : i])
            R_vals.append(R)
            psi_vals.append(psi)
        R_cur = R_vals[-1] if R_vals else 0.0
        psi_cur = psi_vals[-1] if psi_vals else 0.0
        return KuramotoResult(R=R_cur, psi=psi_cur, phases=phases, timeframe=timeframe, window_size=window)

    def _consensus(self, results: Dict[TimeFrame, KuramotoResult]) -> float:
        if not results:
            return 0.0
        weights = {
            TimeFrame.M1: 0.1,
            TimeFrame.M5: 0.2,
            TimeFrame.M15: 0.3,
            TimeFrame.H1: 0.4,
        }
        ws: List[float] = []
        vs: List[float] = []
        for tf, res in results.items():
            w = weights.get(tf, 0.25)
            ws.append(w)
            vs.append(res.R)
        ws_arr = np.asarray(ws)
        vs_arr = np.asarray(vs)
        return float((ws_arr * vs_arr).sum() / ws_arr.sum())

    def _dominant(self, results: Dict[TimeFrame, KuramotoResult]) -> Optional[TimeFrame]:
        if not results:
            return None
        return max(results.items(), key=lambda kv: kv[1].R)[0]

    def _coherence(self, results: Dict[TimeFrame, KuramotoResult]) -> float:
        if len(results) < 2:
            return 1.0
        arr = np.array([r.R for r in results.values()], dtype=float)
        mean = arr.mean()
        if mean == 0:
            return 0.0
        cv = arr.std() / mean
        return float(np.exp(-cv))

    def _select_window(self, prices: np.ndarray) -> int:
        if not self.use_adaptive_window:
            return self.base_window
        if self._window_selector is None:
            return dominant_period_autocorr(prices)
        return int(self._window_selector.select_window(prices))

    def analyze(self, df: pd.DataFrame, price_col: str = "close") -> MultiScaleResult:
        if df.empty or price_col not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column and not be empty")

        prices = df[price_col].astype(float).values
        window = self._select_window(prices)

        tf_results: Dict[TimeFrame, KuramotoResult] = {}
        skipped: List[TimeFrame] = []
        for tf in self.timeframes:
            s = self._resample_close(df, tf, price_col)
            if len(s) < window + 5:
                skipped.append(tf)
                continue
            tf_results[tf] = self.analyze_single_timeframe(s.values, tf, window)

        consensus = self._consensus(tf_results)
        dom = self._dominant(tf_results)
        coh = self._coherence(tf_results)
        return MultiScaleResult(
            consensus_R=consensus,
            timeframe_results=tf_results,
            dominant_scale=dom,
            cross_scale_coherence=coh,
            adaptive_window=window,
            skipped_timeframes=skipped,
        )


# -------------------- Feature adapter --------------------
class MultiScaleKuramotoFeature(BaseFeature):
    """Expose multi-scale Kuramoto consensus as a feature."""

    def __init__(self, analyzer: Optional[MultiScaleKuramoto] = None, *, name: str | None = None) -> None:
        super().__init__(name or "multiscale_kuramoto")
        self._analyzer = analyzer or MultiScaleKuramoto()

    def transform(self, data: pd.DataFrame, *, price_col: str = "close", **_: object) -> FeatureResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MultiScaleKuramotoFeature expects a pandas DataFrame")
        result = self._analyzer.analyze(data, price_col=price_col)

        metadata: Dict[str, float | int | str | List[str]] = {
            "adaptive_window": result.adaptive_window,
            "cross_scale_coherence": result.cross_scale_coherence,
            "timeframes": [str(tf) for tf in result.timeframe_results],
        }
        if result.dominant_scale is not None:
            metadata["dominant_scale"] = str(result.dominant_scale)
        if result.skipped_timeframes:
            metadata["skipped_timeframes"] = [str(tf) for tf in result.skipped_timeframes]
        for tf, tf_result in result.timeframe_results.items():
            metadata[f"R_{tf}"] = tf_result.R
            metadata[f"psi_{tf}"] = tf_result.psi

        return FeatureResult(name=self.name, value=result.consensus_R, metadata=metadata)


__all__ = [
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "TimeFrame",
    "WaveletWindowSelector",
]
