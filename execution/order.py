# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Order:
    side: str  # 'buy' or 'sell'
    qty: float
    price: float | None = None
    type: str = "market"  # 'market'|'limit'

def position_sizing(balance: float, risk: float, price: float, *, max_leverage: float = 5.0) -> float:
    """Risk-aware position size expressed in base units."""

    if price <= 0:
        raise ValueError("price must be positive")
    risk = max(0.0, min(risk, 1.0))
    notional = balance * risk
    qty = notional / price
    leverage_cap = (balance * max_leverage) / price
    return float(max(0.0, min(qty, leverage_cap)))
