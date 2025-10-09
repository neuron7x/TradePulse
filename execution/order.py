# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from dataclasses import dataclass


_NEXTAFTER_BACKOFF_STEPS = 12


@dataclass
class Order:
    side: str  # 'buy' or 'sell'
    qty: float
    price: float | None = None
    type: str = "market"  # 'market'|'limit'


def _normalise_risk(risk: float) -> float:
    """Clamp a potentially ill-conditioned risk fraction into [0, 1]."""

    if not math.isfinite(risk):
        return 0.0
    return max(0.0, min(risk, 1.0))


def _constrain_to_budget(qty: float, price: float, budget: float) -> float:
    """Ensure the quantity never exceeds the notional ``budget``."""

    if qty <= 0.0 or budget <= 0.0:
        return 0.0

    if not math.isfinite(qty):
        qty = budget / price
        if not math.isfinite(qty):
            return 0.0

    max_qty = budget / price
    if math.isfinite(max_qty):
        qty = min(qty, max_qty)

    # ``math.nextafter`` backs the quantity away from zero in discrete
    # floating-point steps.  We limit the loop to a small constant to avoid
    # pathological behaviour while still guaranteeing budget adherence.
    for _ in range(_NEXTAFTER_BACKOFF_STEPS):
        notional = qty * price
        if math.isfinite(notional) and notional <= budget:
            break
        qty = math.nextafter(qty, 0.0)
        if qty <= 0.0:
            return 0.0
    else:
        qty = math.nextafter(qty, 0.0)

    notional = qty * price
    if qty > 0.0 and (not math.isfinite(notional) or notional > budget):
        qty = math.nextafter(qty, 0.0)

    return max(0.0, qty)


def position_sizing(balance: float, risk: float, price: float, *, max_leverage: float = 5.0) -> float:
    """Risk-aware position size expressed in base units."""

    if not math.isfinite(price) or price <= 0.0:
        raise ValueError("price must be positive")
    if not math.isfinite(max_leverage) or max_leverage <= 0.0:
        raise ValueError("max_leverage must be positive")
    if not math.isfinite(balance) or balance <= 0.0:
        return 0.0

    risk = _normalise_risk(risk)
    notional = balance * risk
    if notional <= 0.0 or not math.isfinite(notional):
        return 0.0

    leverage_notional = balance * max_leverage
    if not math.isfinite(leverage_notional):
        leverage_notional = float("inf")

    budget = min(notional, leverage_notional)
    if budget <= 0.0:
        return 0.0

    risk_qty = notional / price
    leverage_cap = leverage_notional / price
    qty = min(risk_qty, leverage_cap)

    return float(_constrain_to_budget(qty, price, budget))
