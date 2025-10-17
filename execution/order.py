# SPDX-License-Identifier: MIT
"""Order sizing utilities anchored in governance-aligned risk budgeting.

The sizing rules here implement the capital allocation policy referenced in
``docs/execution.md`` and the risk controls catalogue in
``docs/risk_ml_observability.md``. The default :class:`RiskAwarePositionSizer`
respects leverage caps, fractional risk budgets, and floating-point safety
margins, ensuring orders remain within the guardrails mandated by the
documentation governance framework.
"""

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
        ) -> float: ...


class RiskAwarePositionSizer(PositionSizer):
    """Default implementation of :class:`interfaces.execution.PositionSizer`.

    The sizer follows the risk-budgeting blueprint in ``docs/execution.md`` and
    the governance safeguards recorded in ``docs/documentation_governance.md``.
    """

    def size(
        self,
        balance: float,
        risk: float,
        price: float,
        *,
        max_leverage: float = 5.0,
    ) -> float:
        """Allocate position size based on balance, risk budget, and leverage.

        Args:
            balance: Available capital in account currency.
            risk: Fraction of capital to deploy (``0``–``1``). Values outside the
                range are clipped to comply with ``docs/execution.md`` guidance.
            price: Execution price of the instrument. Must be positive.
            max_leverage: Maximum allowable leverage multiplier.

        Returns:
            Quantity in base units that satisfies the risk budget while observing
            the leverage ceiling.

        Raises:
            ValueError: If ``price`` is non-positive.

        Examples:
            >>> sizer = RiskAwarePositionSizer()
            >>> round(sizer.size(balance=10_000, risk=0.02, price=25_000), 6)
            0.008

        Notes:
            Subnormal floating-point values can violate the risk budget after
            multiplication. The implementation iteratively nudges the quantity
            toward zero using :func:`math.nextafter` until it adheres to the
            budget, satisfying the precision safeguards in
            ``docs/documentation_governance.md``.
        """
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


def position_sizing(
    balance: float,
    risk: float,
    price: float,
    *,
    max_leverage: float = 5.0,
) -> float:
    """Convenience wrapper returning risk-aware position size in base units.

    Args:
        balance: Available capital in account currency.
        risk: Fraction of capital to deploy.
        price: Execution price of the instrument.
        max_leverage: Maximum allowable leverage multiplier.

    Returns:
        float: Quantity computed via :class:`RiskAwarePositionSizer`.
    """

    return RiskAwarePositionSizer().size(
        balance,
        risk,
        price,
        max_leverage=max_leverage,
    )


__all__ = ["Order", "RiskAwarePositionSizer", "position_sizing"]
