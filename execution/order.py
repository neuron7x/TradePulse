# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Order:
    side: str  # 'buy' or 'sell'
    qty: float
    price: float | None = None
    type: str = "market"  # 'market'|'limit'

def position_sizing(balance: float, risk: float, price: float) -> float:
    # Kelly-like simple sizing
    return max(0.0, min(balance * risk / max(price,1e-9), balance/price))
