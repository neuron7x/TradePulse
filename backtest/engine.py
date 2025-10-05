# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class Result:
    pnl: float
    max_dd: float
    trades: int

def walk_forward(prices: np.ndarray, signal_fn: Callable[[np.ndarray], np.ndarray], fee: float=0.0005) -> Result:
    sig = signal_fn(prices)  # -1/0/1
    pos = 0
    pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    for t in range(1, len(prices)):
        if sig[t] != pos:
            trades += 1
            pnl -= fee * abs(sig[t]-pos)
            pos = sig[t]
        pnl += pos * (prices[t] - prices[t-1])
        peak = max(peak, pnl)
        max_dd = min(max_dd, pnl - peak)
    return Result(pnl=pnl, max_dd=max_dd, trades=trades)
