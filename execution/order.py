# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from typing import Protocol

from domain import Order

try:  # pragma: no cover - optional dependency boundary
    from interfaces.execution import PositionSizer
except ModuleNotFoundError:  # pragma: no cover
    class PositionSizer(Protocol):
        def size(
            self,
            balance: float,
            risk: float,
            price: float,
            *,
            max_leverage: float = 5.0,
        ) -> float:
            ...


class RiskAwarePositionSizer(PositionSizer):
    """Default implementation of :class:`interfaces.execution.PositionSizer`."""

    def size(
        self,
        balance: float,
        risk: float,
        price: float,
        *,
        max_leverage: float = 5.0,
    ) -> float:
        if price <= 0:
            raise ValueError("price must be positive")
        risk = max(0.0, min(risk, 1.0))
        notional = balance * risk
        if notional <= 0.0:
            return 0.0

        risk_qty = notional / price
        leverage_cap = (balance * max_leverage) / price
        qty = min(risk_qty, leverage_cap)

        if qty > 0.0 and qty * price > notional:
            # When working with denormals the round-trip multiplication can
            # overshoot the risk budget due to floating point rounding.
            # Bias the quantity towards zero until it fits within the budget.
            qty = math.nextafter(qty, 0.0)
            while qty > 0.0 and qty * price > notional:
                qty = math.nextafter(qty, 0.0)

        return float(max(0.0, qty))


def position_sizing(balance: float, risk: float, price: float, *, max_leverage: float = 5.0) -> float:
    """Risk-aware position size expressed in base units."""

    return RiskAwarePositionSizer().size(
        balance,
        risk,
        price,
        max_leverage=max_leverage,
    )


__all__ = ["Order", "RiskAwarePositionSizer", "position_sizing"]
