
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -------------------- Lightweight Hilbert (no SciPy) --------------------
def analytic_signal(x: np.ndarray) -> np.ndarray:
    """
    Compute analytic signal via FFT-based Hilbert transform.
    Returns complex signal: x + i * H(x)
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    X = np.fft.fft(x, n=N)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = 1.0
        h[N//2] = 1.0
        h[1:N//2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N+1)//2] = 2.0
    xa = np.fft.ifft(X * h)
    return xa

def extract_phase(x: np.ndarray) -> np.ndarray:
    """Detrend (simple linear) + analytic signal -> angle"""
    x = np.asarray(x, dtype=float)
    n = x.size
    t = np.arange(n)
    # simple linear detrend
    A = np.vstack([t, np.ones(n)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    detrended = x - (m*t + b)
    z = analytic_signal(detrended)
    return np.angle(z)

# -------------------- Adaptive window via autocorrelation --------------------
def dominant_period_autocorr(x: np.ndarray, min_window: int = 50, max_window: int = 500) -> int:
    """
    Estimate dominant period using (biased) autocorrelation peak.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n < min_window * 2:
        return max(min_window, min(n//4, max_window))

    # FFT-based autocorr
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f * np.conjugate(f)).real[:n]
    ac = ac / ac[0] if ac[0] != 0 else ac

    # find first local maximum after lag 1
    start = max(2, min_window//4)
    end = min(n//2, max_window*2)
    if end <= start+1:
        return min_window
    lag = start + np.argmax(ac[start:end])
    win = int(np.clip(lag*2, min_window, max_window))
    return win

# -------------------- Data models --------------------
@dataclass
class KuramotoResult:
    R: float
    psi: float
    phases: np.ndarray
    timeframe_sec: int
    window_size: int

@dataclass
class MultiScaleResult:
    consensus_R: float
    timeframe_results: Dict[int, KuramotoResult]
    dominant_scale_sec: Optional[int]
    cross_scale_coherence: float
    adaptive_window: int

# -------------------- Core class --------------------
class MultiScaleKuramoto:
    """
    Multi-scale Kuramoto analysis with zero external deps (NumPy/Pandas only).
    timeframes: list of seconds (e.g., [60, 300, 900, 3600])
    """
    def __init__(self, timeframes: Optional[List[int]] = None, use_adaptive_window: bool = True, base_window: int = 200):
        self.timeframes = timeframes or [60, 300, 900, 3600]
        self.use_adaptive_window = use_adaptive_window
        self.base_window = base_window

    @staticmethod
    def _kuramoto(phases: np.ndarray) -> Tuple[float, float]:
        z = np.exp(1j * phases).mean()
        return float(np.abs(z)), float(np.angle(z))

    def _resample_close(self, df: pd.DataFrame, tf_sec: int, price_col: str) -> pd.Series:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df[price_col].resample(f"{tf_sec}S").last().dropna()

    def analyze_single_timeframe(self, prices: np.ndarray, tf_sec: int, window: int) -> KuramotoResult:
        phases = extract_phase(prices)
        R_vals = []
        psi_vals = []
        for i in range(window, len(phases)):
            R, psi = self._kuramoto(phases[i-window:i])
            R_vals.append(R); psi_vals.append(psi)
        R_cur = R_vals[-1] if R_vals else 0.0
        psi_cur = psi_vals[-1] if psi_vals else 0.0
        return KuramotoResult(R=R_cur, psi=psi_cur, phases=phases, timeframe_sec=tf_sec, window_size=window)

    def _consensus(self, results: Dict[int, KuramotoResult]) -> float:
        if not results:
            return 0.0
        # weights favor higher timeframes
        weights = {60:0.1, 300:0.2, 900:0.3, 3600:0.4}
        ws, vs = [], []
        for tf, res in results.items():
            w = weights.get(tf, 0.25)
            ws.append(w); vs.append(res.R)
        ws = np.asarray(ws); vs = np.asarray(vs)
        return float((ws*vs).sum() / ws.sum())

    def _dominant(self, results: Dict[int, KuramotoResult]) -> Optional[int]:
        if not results: return None
        return max(results.items(), key=lambda kv: kv[1].R)[0]

    def _coherence(self, results: Dict[int, KuramotoResult]) -> float:
        if len(results) < 2: return 1.0
        arr = np.array([r.R for r in results.values()], dtype=float)
        mean = arr.mean()
        if mean == 0: return 0.0
        cv = arr.std() / mean
        return float(np.exp(-cv))

    def analyze(self, df: pd.DataFrame, price_col: str = "close") -> MultiScaleResult:
        if df.empty or price_col not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column and not be empty")

        prices = df[price_col].astype(float).values
        window = dominant_period_autocorr(prices) if self.use_adaptive_window else self.base_window

        tf_results: Dict[int, KuramotoResult] = {}
        for tf in self.timeframes:
            s = self._resample_close(df, tf, price_col)
            if len(s) < window + 5:  # need enough data
                continue
            tf_results[tf] = self.analyze_single_timeframe(s.values, tf, window)

        consensus = self._consensus(tf_results)
        dom = self._dominant(tf_results)
        coh = self._coherence(tf_results)
        return MultiScaleResult(consensus_R=consensus, timeframe_results=tf_results, dominant_scale_sec=dom, cross_scale_coherence=coh, adaptive_window=window)
