# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from core.utils.metrics import get_metrics_collector

@dataclass
class Result:
    pnl: float
    max_dd: float
    trades: int

def walk_forward(prices: np.ndarray, signal_fn: Callable[[np.ndarray], np.ndarray], fee: float = 0.0005, initial_capital: float = 0.0, strategy_name: str = "default") -> Result:
    """Vectorised walk-forward backtest with risk-aware bookkeeping."""

    metrics = get_metrics_collector()
    
    with metrics.measure_backtest(strategy_name) as ctx:
        price_array = np.asarray(prices, dtype=float)
        if price_array.ndim != 1 or price_array.size < 2:
            raise ValueError("prices must be a 1-D array with at least two observations")

        signals = np.asarray(signal_fn(price_array), dtype=float)
        if signals.shape != price_array.shape:
            raise ValueError("signal_fn must return an array with the same length as prices")

        signals = np.clip(signals, -1.0, 1.0)
        price_moves = np.diff(price_array)
        positions = signals[1:]
        prev_positions = np.concatenate([[0.0], positions[:-1]])
        position_changes = positions - prev_positions
        trade_costs = np.abs(position_changes) * fee
        pnl = positions * price_moves - trade_costs

        equity_curve = np.cumsum(pnl) + initial_capital
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - peaks
        pnl_total = float(pnl.sum())
        max_dd = float(drawdowns.min()) if drawdowns.size else 0.0
        trades = int(np.count_nonzero(position_changes))
        
        # Record metrics
        ctx["pnl"] = pnl_total
        ctx["max_dd"] = max_dd
        ctx["trades"] = trades
        
        return Result(pnl=pnl_total, max_dd=max_dd, trades=trades)

