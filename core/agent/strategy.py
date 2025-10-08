# SPDX-License-Identifier: MIT
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pandas as pd

@dataclass(slots=True)
class StrategyDiagnostics:
    """Snapshot of diagnostic metrics captured during the last simulation."""

    equity_curve: list[float]
    positions: list[float]
    pnl: list[float]
    max_drawdown: float
    max_drawdown_pct: float
    trades: int
    sharpe: float
    terminal_value: float
    exposure: float
    turnover: float
    hit_rate: float
    average_gain: float
    average_loss: float

    def as_params(self) -> Dict[str, Any]:
        """Return a dictionary compatible with ``Strategy.params`` updates."""

        return {
            "last_equity_curve": self.equity_curve,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trades": self.trades,
            "sharpe": self.sharpe,
            "terminal_value": self.terminal_value,
            "exposure": self.exposure,
            "turnover": self.turnover,
            "hit_rate": self.hit_rate,
            "average_gain": self.average_gain,
            "average_loss": self.average_loss,
        }

    @staticmethod
    def _safe_mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else 0.0

    @classmethod
    def from_arrays(
        cls,
        equity_curve: np.ndarray,
        positions: np.ndarray,
        pnl: np.ndarray,
    ) -> "StrategyDiagnostics":
        equity_curve = np.asarray(equity_curve, dtype=float)
        positions = np.asarray(positions, dtype=float)
        pnl = np.asarray(pnl, dtype=float)

        if equity_curve.size:
            peak = np.maximum.accumulate(np.concatenate(([0.0], equity_curve)))[1:]
            drawdown = equity_curve - peak
            max_drawdown = float(drawdown.min()) if drawdown.size else 0.0
            denom = np.where(np.abs(peak) < 1e-9, 1.0, np.abs(peak))
            max_drawdown_pct = float(
                np.max((peak - equity_curve) / denom) if denom.size else 0.0
            )
            terminal_value = float(equity_curve[-1])
        else:
            peak = np.array([], dtype=float)
            drawdown = np.array([], dtype=float)
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
            terminal_value = 0.0

        trades = int(np.count_nonzero(np.diff(positions))) if positions.size else 0
        std = float(pnl.std()) if pnl.size else 0.0
        sharpe = float(pnl.mean() / (std + 1e-9)) if pnl.size else 0.0
        exposure = float(np.mean(np.abs(positions))) if positions.size else 0.0
        turnover = (
            float(np.sum(np.abs(np.diff(positions)))) if positions.size > 1 else 0.0
        )
        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        total_trades = wins.size + losses.size
        hit_rate = float(wins.size / total_trades) if total_trades else 0.0
        average_gain = cls._safe_mean(wins)
        average_loss = cls._safe_mean(losses)

        return cls(
            equity_curve=equity_curve.tolist(),
            positions=positions.tolist(),
            pnl=pnl.tolist(),
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            sharpe=sharpe,
            terminal_value=terminal_value,
            exposure=exposure,
            turnover=turnover,
            hit_rate=hit_rate,
            average_gain=average_gain,
            average_loss=average_loss,
        )


@dataclass
class Strategy:
    name: str
    params: Dict[str, Any]
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    diagnostics: StrategyDiagnostics | None = field(default=None, init=False, repr=False)

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

        def _update_diagnostics(
            equity_curve: np.ndarray, positions: np.ndarray, pnl: np.ndarray
        ) -> None:
            diagnostics = StrategyDiagnostics.from_arrays(equity_curve, positions, pnl)
            self.diagnostics = diagnostics
            self.params.update(diagnostics.as_params())

        if data is None:
            series = pd.Series(np.linspace(100.0, 101.0, 256), dtype=float)
        else:
            series = _to_price_series(data)

        series = series.astype(float)
        if isinstance(series.index, pd.DatetimeIndex):
            series = series[~series.index.duplicated(keep="last")].sort_index()
        series = series.replace([np.inf, -np.inf], np.nan)
        if series.isna().all():
            self.score = 0.0
            empty = np.array([], dtype=float)
            _update_diagnostics(empty, empty, empty)
            return self.score
        series = series.ffill().bfill()

        returns = series.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            self.score = 0.0
            empty = np.array([], dtype=float)
            _update_diagnostics(empty, empty, empty)
            return self.score

        lookback = int(self.params.get("lookback", 20))
        threshold = float(self.params.get("threshold", 0.5))
        risk_budget = float(self.params.get("risk_budget", 1.0))
        effective_lookback = max(1, min(lookback, len(returns)))

        rolling_mean = returns.rolling(window=effective_lookback, min_periods=effective_lookback).mean().fillna(0.0)
        rolling_vol = (
            returns.rolling(window=effective_lookback, min_periods=1)
            .std(ddof=0)
            .replace(0.0, np.nan)
            .ffill()
            .bfill()
            .fillna(1e-6)
        )
        zscore = (rolling_mean / rolling_vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        signal = np.where(zscore > threshold, -1.0, np.where(zscore < -threshold, 1.0, 0.0))
        if signal.size:
            position = np.concatenate(([0.0], signal[:-1])) * risk_budget
        else:
            position = np.array([], dtype=float)

        pnl = position * returns.to_numpy()
        equity = np.cumsum(pnl)
        _update_diagnostics(equity, position, pnl)
        if equity.size == 0:
            self.score = 0.0
            return self.score

        sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9)
        terminal = equity[-1]
        raw_score = terminal + 0.5 * sharpe
        self.score = float(np.clip(raw_score, -1.0, 2.0))
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
