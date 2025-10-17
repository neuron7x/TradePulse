# SPDX-License-Identifier: MIT
"""Order sizing utilities anchored in governance-aligned risk budgeting.

The sizing rules here implement the capital allocation policy referenced in
``docs/execution.md`` and the risk controls catalogue in
``docs/risk_ml_observability.md``. The default :class:`RiskAwarePositionSizer`
respects leverage caps, fractional risk budgets, and floating-point safety
margins, ensuring orders remain within the guardrails mandated by the
documentation governance framework.

The module also exposes :class:`ConstrainedPositionSizer`, a higher-order
implementation that blends Kelly fraction budgeting, CPPI capital preservation,
drawdown-aware throttling, and volatility caps. The class is designed for
portfolio-level risk harmonisation: it consumes portfolio state snapshots and
returns automatically adjusted order deltas that honour the configured
constraints. This makes it suitable for the scenarios documented in
``docs/risk_ml_observability.md`` where Kelly utilisation, drawdown tripwires,
and volatility overlays must be coordinated in real time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol

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


@dataclass(slots=True)
class PositionSizingConstraints:
    """Risk guardrails enforced by :class:`ConstrainedPositionSizer`."""

    max_drawdown: float = 0.15
    max_portfolio_volatility: float = 0.35
    kelly_fraction_limit: float = 0.4
    cppi_multiplier: float = 3.0
    cppi_floor: float = 0.8
    volatility_buffer: float = 0.02
    min_order_size: float = 0.0
    max_order_size: float | None = None
    max_leverage: float = 5.0

    def __post_init__(self) -> None:
        if self.max_drawdown < 0.0:
            raise ValueError("max_drawdown must be non-negative")
        if self.max_portfolio_volatility <= 0.0:
            raise ValueError("max_portfolio_volatility must be positive")
        if self.kelly_fraction_limit <= 0.0:
            raise ValueError("kelly_fraction_limit must be positive")
        if self.cppi_multiplier <= 0.0:
            raise ValueError("cppi_multiplier must be positive")
        if not 0.0 <= self.cppi_floor <= 1.0:
            raise ValueError("cppi_floor must be between 0 and 1")
        if self.volatility_buffer < 0.0:
            raise ValueError("volatility_buffer must be non-negative")
        if self.min_order_size < 0.0:
            raise ValueError("min_order_size must be non-negative")
        if self.max_order_size is not None and self.max_order_size <= 0.0:
            raise ValueError("max_order_size must be positive when specified")
        if self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be positive")


@dataclass(slots=True)
class PortfolioState:
    """Snapshot of the portfolio used for constrained sizing."""

    balance: float
    equity: float
    peak_equity: float
    volatility: float
    positions: Mapping[str, float] = field(default_factory=dict)
    risk_budgets: Mapping[str, float] | None = None
    risk_exposures: Mapping[str, float] | None = None

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def position_for(self, symbol: str) -> float:
        return float(self.positions.get(symbol, 0.0))

    def risk_budget_for(self, symbol: str) -> float | None:
        if self.risk_budgets is None:
            return None
        value = self.risk_budgets.get(symbol)
        return None if value is None else float(value)

    def risk_exposure_for(self, symbol: str) -> float:
        if self.risk_exposures is None:
            return 0.0
        return float(self.risk_exposures.get(symbol, 0.0))


@dataclass(slots=True)
class PositionSizingRequest:
    """Inputs describing the desired allocation for a single instrument."""

    symbol: str
    direction: int
    price: float
    risk_fraction: float
    confidence: float = 1.0
    forecast_edge: float | None = None
    forecast_variance: float | None = None
    instrument_volatility: float | None = None
    min_trade_qty: float = 0.0
    max_trade_qty: float | None = None
    leverage_limit: float | None = None


@dataclass(slots=True)
class PositionSizingResult:
    """Result of constrained sizing with diagnostics."""

    order_quantity: float
    target_position: float
    applied_fraction: float
    notes: dict[str, float] = field(default_factory=dict)


class ConstrainedPositionSizer(RiskAwarePositionSizer):
    """Position sizer that enforces portfolio-level constraints."""

    def __init__(
        self,
        constraints: PositionSizingConstraints | None = None,
    ) -> None:
        super().__init__()
        self._constraints = constraints or PositionSizingConstraints()

    def size(
        self,
        balance: float,
        risk: float,
        price: float,
        *,
        max_leverage: float = 5.0,
    ) -> float:
        state = PortfolioState(
            balance=balance,
            equity=balance,
            peak_equity=max(balance, 0.0),
            volatility=0.0,
        )
        direction = 1 if risk >= 0.0 else -1
        request = PositionSizingRequest(
            symbol="__anonymous__",
            direction=direction,
            price=price,
            risk_fraction=abs(risk),
            leverage_limit=max_leverage,
        )
        result = self.size_order(request, state)
        return abs(result.target_position)

    def size_order(
        self,
        request: PositionSizingRequest,
        state: PortfolioState,
    ) -> PositionSizingResult:
        if request.price <= 0.0:
            raise ValueError("price must be positive")

        capital = max(state.equity, 0.0)
        if capital == 0.0:
            return PositionSizingResult(
                order_quantity=0.0,
                target_position=state.position_for(request.symbol),
                applied_fraction=0.0,
                notes={"capital": 0.0},
            )

        direction = 0
        if request.direction > 0:
            direction = 1
        elif request.direction < 0:
            direction = -1

        if direction == 0:
            return PositionSizingResult(
                order_quantity=0.0,
                target_position=state.position_for(request.symbol),
                applied_fraction=0.0,
                notes={"direction": 0.0},
            )

        constraints = self._constraints
        notes: dict[str, float] = {}

        risk_fraction = max(0.0, min(request.risk_fraction, 1.0))
        budget = state.risk_budget_for(request.symbol)
        if budget is not None:
            risk_fraction = min(risk_fraction, max(0.0, budget))
            notes["risk_budget"] = budget

        existing_usage = state.risk_exposure_for(request.symbol)
        total_usage = existing_usage
        if state.risk_exposures is not None:
            total_usage = sum(max(0.0, value) for value in state.risk_exposures.values())
            residual = max(0.0, 1.0 - (total_usage - max(0.0, existing_usage)))
            risk_fraction = min(risk_fraction, residual)
            notes["portfolio_residual"] = residual

        confidence = max(0.0, min(request.confidence, 1.0))
        kelly_fraction: float | None = None
        if request.forecast_edge is not None and request.forecast_variance is not None:
            variance = request.forecast_variance
            if variance > 0.0:
                kelly_fraction = (request.forecast_edge / variance) * confidence
                kelly_fraction = max(
                    -constraints.kelly_fraction_limit,
                    min(kelly_fraction, constraints.kelly_fraction_limit),
                )
                notes["kelly_fraction"] = kelly_fraction
            else:
                notes["kelly_fraction"] = 0.0

        candidate_fraction = min(risk_fraction, constraints.kelly_fraction_limit) * direction
        if kelly_fraction is not None:
            kelly_direction = 1 if kelly_fraction >= 0.0 else -1
            if direction != kelly_direction:
                return PositionSizingResult(
                    order_quantity=0.0,
                    target_position=state.position_for(request.symbol),
                    applied_fraction=0.0,
                    notes={"kelly_conflict": 1.0},
                )
            candidate_fraction = direction * min(
                min(risk_fraction, abs(kelly_fraction)),
                constraints.kelly_fraction_limit,
            )

        drawdown = state.drawdown
        notes["drawdown"] = drawdown
        if constraints.max_drawdown > 0.0 and drawdown >= constraints.max_drawdown:
            return PositionSizingResult(
                order_quantity=0.0,
                target_position=state.position_for(request.symbol),
                applied_fraction=0.0,
                notes={"drawdown": drawdown},
            )

        if constraints.max_drawdown > 0.0 and drawdown > 0.0:
            scale = max(0.0, 1.0 - drawdown / constraints.max_drawdown)
            candidate_fraction *= scale
            notes["drawdown_scale"] = scale

        leverage_cap = min(constraints.max_leverage, request.leverage_limit or constraints.max_leverage)
        cppi_limit = self._cppi_fraction(state, leverage_cap)
        notes["cppi_limit"] = cppi_limit
        if cppi_limit <= 0.0:
            return PositionSizingResult(
                order_quantity=0.0,
                target_position=state.position_for(request.symbol),
                applied_fraction=0.0,
                notes={"cppi_limit": cppi_limit},
            )

        candidate_fraction = max(-cppi_limit, min(candidate_fraction, cppi_limit))

        instrument_vol = request.instrument_volatility or 0.0
        if instrument_vol < 0.0:
            instrument_vol = 0.0
        projected = math.hypot(state.volatility, instrument_vol * abs(candidate_fraction))
        vol_limit = max(1e-12, constraints.max_portfolio_volatility - constraints.volatility_buffer)
        if projected > constraints.max_portfolio_volatility:
            scale = constraints.max_portfolio_volatility / projected
            candidate_fraction *= scale
            notes["volatility_scale"] = scale
        elif projected > vol_limit:
            scale = vol_limit / projected
            candidate_fraction *= scale
            notes["volatility_scale"] = scale

        notional = capital * candidate_fraction
        max_notional = capital * leverage_cap
        if abs(notional) > max_notional:
            scale = max_notional / max(abs(notional), 1e-12)
            candidate_fraction *= scale
            notional = capital * candidate_fraction
            notes["leverage_scale"] = scale

        target_position = notional / request.price
        existing_position = state.position_for(request.symbol)
        order_quantity = target_position - existing_position

        min_size = max(constraints.min_order_size, request.min_trade_qty)
        if abs(order_quantity) < min_size:
            order_quantity = 0.0
            target_position = existing_position

        max_size = constraints.max_order_size
        if request.max_trade_qty is not None:
            max_size = request.max_trade_qty if max_size is None else min(max_size, request.max_trade_qty)

        if max_size is not None and abs(order_quantity) > max_size:
            order_quantity = math.copysign(max_size, order_quantity)
            target_position = existing_position + order_quantity
            notes["order_clip"] = max_size

        applied_fraction = 0.0
        if capital > 0.0:
            applied_fraction = (target_position * request.price) / capital

        notes["final_fraction"] = applied_fraction

        return PositionSizingResult(
            order_quantity=order_quantity,
            target_position=target_position,
            applied_fraction=applied_fraction,
            notes=notes,
        )

    def _cppi_fraction(self, state: PortfolioState, leverage_cap: float) -> float:
        if state.equity <= 0.0 or state.peak_equity <= 0.0:
            return 0.0
        floor_value = self._constraints.cppi_floor * state.peak_equity
        cushion = max(0.0, state.equity - floor_value)
        if cushion <= 0.0:
            return 0.0
        limit = (self._constraints.cppi_multiplier * cushion) / max(state.equity, 1e-12)
        return min(limit, leverage_cap)

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


__all__ = [
    "Order",
    "RiskAwarePositionSizer",
    "PositionSizingConstraints",
    "PortfolioState",
    "PositionSizingRequest",
    "PositionSizingResult",
    "ConstrainedPositionSizer",
    "position_sizing",
]
