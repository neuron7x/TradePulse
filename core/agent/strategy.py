# SPDX-License-Identifier: MIT
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

@dataclass
class Strategy:
    name: str
    params: Dict[str, Any]
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def generate_mutation(self, *, scale: float = 0.1) -> "Strategy":
        rng = np.random.default_rng()
        new_params: Dict[str, Any] = {}
        for key, value in self.params.items():
            if isinstance(value, (int, float)):
                perturb = rng.normal(loc=0.0, scale=abs(value) * scale + 1e-6)
                candidate = value + perturb
                if isinstance(value, int):
                    candidate = max(1, int(round(candidate)))
                new_params[key] = candidate
            else:
                new_params[key] = value
        mutated = Strategy(name=f"{self.name}_mut", params=new_params)
        mutated.validate_params()
        return mutated

    def validate_params(self) -> None:
        lookback = int(self.params.get("lookback", 20))
        self.params["lookback"] = max(5, min(lookback, 500))
        threshold = float(self.params.get("threshold", 0.0))
        self.params["threshold"] = max(0.0, min(threshold, 5.0))
        risk = float(self.params.get("risk_budget", 1.0))
        self.params["risk_budget"] = max(0.01, min(risk, 10.0))

    def simulate_performance(self, data: Any) -> float:
        """Deterministic walk-forward score using rolling mean-reversion logic."""

        self.validate_params()
        if data is None:
            series = pd.Series(np.linspace(100.0, 101.0, 256), dtype=float)
        else:
            series = _to_price_series(data)
        returns = series.pct_change().dropna()
        lookback = int(self.params.get("lookback", 20))
        threshold = float(self.params.get("threshold", 0.5))
        risk_budget = float(self.params.get("risk_budget", 1.0))

        rolling_mean = returns.rolling(window=lookback).mean().fillna(0.0)
        rolling_vol = returns.rolling(window=lookback).std(ddof=0).replace(0, np.nan).bfill().fillna(1e-6)
        zscore = rolling_mean / rolling_vol
        signal = np.where(zscore > threshold, -1, np.where(zscore < -threshold, 1, 0))
        position = np.concatenate([[0], signal[:-1]]) * risk_budget
        pnl = position * returns.to_numpy()
        equity = np.cumsum(pnl)
        if equity.size == 0:
            self.score = 0.0
            return self.score
        peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        drawdown = equity - peak
        sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9)
        terminal = equity[-1]
        raw_score = terminal + 0.5 * sharpe
        self.score = float(np.clip(raw_score, -1.0, 2.0))
        self.params["last_equity_curve"] = equity.tolist()
        self.params["max_drawdown"] = float(drawdown.min())
        self.params["trades"] = int(np.count_nonzero(np.diff(position)))
        return self.score

@dataclass
class PiAgent:
    strategy: Strategy
    hysteresis: float = 0.05

    def __post_init__(self) -> None:
        self._instability_score: float = 0.0
        self._cooldown: int = 0
        self.strategy.validate_params()

    def detect_instability(self, market_state: Dict[str, float]) -> bool:
        R = market_state.get("R", 0.0)
        dH = market_state.get("delta_H", 0.0)
        kappa = market_state.get("kappa_mean", 0.0)
        transition = market_state.get("transition_score", 0.0)
        hard_trigger = R > 0.75 and dH < 0 and kappa < 0
        score = 0.6 * max(R - 0.7, 0.0) + 0.25 * max(-dH, 0.0) + 0.15 * max(-kappa, 0.0) + 0.2 * transition
        self._instability_score = 0.7 * self._instability_score + 0.3 * score
        threshold = self.strategy.params.get("instability_threshold", 0.2)
        triggered = (hard_trigger or self._instability_score > threshold) and self._cooldown == 0
        if triggered:
            self._cooldown = 3
        elif self._cooldown > 0:
            self._cooldown -= 1
        return triggered

    def mutate(self) -> "PiAgent":
        return PiAgent(strategy=self.strategy.generate_mutation())

    def repair(self) -> None:
        for key, value in list(self.strategy.params.items()):
            if isinstance(value, (int, float)) and math.isnan(value):
                self.strategy.params[key] = 0.0
        self.strategy.validate_params()

    def evaluate_and_adapt(self, market_state) -> str:
        action = "hold"
        if self.detect_instability(market_state):
            action = "enter"
        elif market_state.get("phase_reversal", False) and self._instability_score < (self.strategy.params.get("instability_threshold", 0.2) - self.hysteresis):
            action = "exit"
        return action


def _to_price_series(data: Any) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float)
    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        return data["close"].astype(float)
    if isinstance(data, (list, tuple, np.ndarray)):
        return pd.Series(np.asarray(data, dtype=float))
    raise TypeError("Unsupported data type for simulate_performance")
