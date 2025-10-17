"""Unified risk manager orchestrating sizing, limits and stops."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from risk.limits import LimitConfig, RiskState
from risk.sizing import aggregate_risk_state, notional_limit, volatility_target_position


@dataclass(slots=True)
class StopPolicy:
    atr_multiple: float = 3.0
    daily_loss_limit: float = 100_000.0


class RiskManager:
    def __init__(self, cfg: LimitConfig, stop_policy: StopPolicy | None = None) -> None:
        self.cfg = cfg
        self.stop_policy = stop_policy or StopPolicy(daily_loss_limit=cfg.max_daily_loss)
        self.state = RiskState()

    def target_vol_position(self, signal_strength: float, sigma: float) -> float:
        return volatility_target_position(
            signal_strength=signal_strength,
            sigma=sigma,
            target_vol=self.cfg.target_vol,
            max_leverage=self.cfg.max_leverage,
        )

    def check_limits(self, state: Dict[str, float]) -> None:
        notional = state.get("notional", 0.0)
        if notional > self.cfg.max_position_notional:
            raise ValueError("Position notional exceeds configured limits")
        pnl = state.get("pnl", 0.0)
        if pnl <= -self.cfg.max_daily_loss:
            raise ValueError("Daily loss limit breached")
        drawdown = state.get("drawdown", 0.0)
        if drawdown >= self.cfg.max_drawdown:
            raise ValueError("Maximum drawdown breached")

    def apply_stops(self, bar: Dict[str, float], position: float) -> Dict[str, float]:
        atr = bar.get("atr")
        price = bar["close"]
        if atr is None:
            return {}
        if position > 0:
            stop_level = price - self.stop_policy.atr_multiple * atr
        else:
            stop_level = price + self.stop_policy.atr_multiple * atr
        return {"stop": stop_level}

    def update_state(self, position: float, price: float, pnl: float, timestamp: datetime) -> None:
        self.state.sigma = self.state.sigma or 0.0
        agg = aggregate_risk_state(position, price, pnl)
        self.state.notional = agg["notional"]
        self.state.pnl = agg["pnl"]
        self.state.timestamp = timestamp.timestamp()

    def clip_position(self, desired_position: float, price: float) -> float:
        return notional_limit(desired_position, price, self.cfg.max_position_notional)

