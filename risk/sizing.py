"""Utility functions for volatility targeting and position sizing."""

from __future__ import annotations

from typing import Dict


def volatility_target_position(signal_strength: float, sigma: float, target_vol: float, max_leverage: float) -> float:
    sigma = max(sigma, 1e-8)
    raw = (target_vol / sigma) * signal_strength
    return max(min(raw, max_leverage), -max_leverage)


def notional_limit(position: float, price: float, max_notional: float) -> float:
    notional = abs(position * price)
    if notional <= max_notional:
        return position
    scale = max_notional / max(notional, 1e-8)
    return position * scale


def aggregate_risk_state(position: float, price: float, pnl: float) -> Dict[str, float]:
    return {
        "notional": abs(position * price),
        "pnl": pnl,
    }

