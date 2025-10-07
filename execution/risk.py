# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Iterable, Mapping


def portfolio_heat(positions: Iterable[Mapping[str, float]]) -> float:
    """Compute aggregate risk heat with directionality and weights."""

    total = 0.0
    for pos in positions:
        qty = float(pos.get("qty", 0.0))
        price = float(pos.get("price", 0.0))
        risk_weight = float(pos.get("risk_weight", 1.0))
        side = pos.get("side", "long")
        direction = 1.0 if side == "long" else -1.0
        total += abs(qty * price * risk_weight * direction)
    return float(total)
