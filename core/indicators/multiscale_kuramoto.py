from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from enum import Enum
from scipy import signal
from typing import Dict, List, Optional, Tuple

class TimeFrame(Enum): M1=60; M5=300; M15=900; H1=3600
@dataclass
class KuramotoResult: R: float; psi: float
@dataclass
class MultiScaleResult: consensus_R: float; cross_scale_coherence: float; dominant_scale: Optional[TimeFrame]; adaptive_window: int; timeframe_results: Dict[TimeFrame, KuramotoResult]
def _hilbert_phase(x): z=signal.hilbert(signal.detrend(np.asarray(x,float))); return np.angle(z)
def _kuramoto(ph): c=np.mean(np.exp(1j*ph)); return float(np.abs(c)), float(np.angle(c))
def analyze_simple(df: pd.DataFrame, price_col: str="close", window:int=128)->MultiScaleResult:
    phases=_hilbert_phase(df[price_col].values); R,psi=_kuramoto(phases[-window:] if phases.size>=window else phases)
    res={TimeFrame.M1: KuramotoResult(R,psi)}
    return MultiScaleResult(R, 1.0, TimeFrame.M1, window, res)
