"""Limit models used by the unified risk manager."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LimitConfig:
    target_vol: float = 0.10
    max_leverage: float = 2.0
    max_position_notional: float = 5_000_000.0
    max_daily_loss: float = 100_000.0
    max_drawdown: float = 0.15


@dataclass(slots=True)
class RiskState:
    sigma: float = 0.0
    pnl: float = 0.0
    notional: float = 0.0
    drawdown: float = 0.0
    timestamp: float = 0.0

