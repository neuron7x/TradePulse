# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping

from interfaces.execution import PortfolioRiskAnalyzer, RiskController

from core.data.catalog import normalize_symbol


class RiskError(RuntimeError):
    """Base exception for risk-control violations."""


class LimitViolation(RiskError):
    """Raised when position or notional limits would be breached."""


class OrderRateExceeded(RiskError):
    """Raised when the order rate throttle blocks a submission."""


@dataclass(slots=True)
class RiskLimits:
    """Risk guardrails that must be enforced prior to execution."""

    max_notional: float = float("inf")
    max_position: float = float("inf")
    max_orders_per_interval: int = 60
    interval_seconds: float = 1.0
    max_daily_loss: float = float("inf")
    soft_loss_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.max_daily_loss < 0:
            object.__setattr__(self, "max_daily_loss", abs(self.max_daily_loss))
        ratio = float(self.soft_loss_ratio)
        if math.isnan(ratio) or math.isinf(ratio):
            ratio = 0.0
        object.__setattr__(self, "soft_loss_ratio", min(max(ratio, 0.0), 1.0))


class KillSwitch:
    """Global kill-switch toggled on critical failures."""

    def __init__(self) -> None:
        self._triggered = False
        self._reason = ""

    def trigger(self, reason: str) -> None:
        self._triggered = True
        self._reason = reason

    def reset(self) -> None:
        self._triggered = False
        self._reason = ""

    @property
    def reason(self) -> str:
        return self._reason

    def is_triggered(self) -> bool:
        return self._triggered

    def guard(self) -> None:
        if self._triggered:
            raise RiskError(f"Kill-switch engaged: {self._reason or 'unspecified reason'}")


class RiskManager(RiskController):
    """Apply notional/position caps and order throttling."""

    def __init__(
        self,
        limits: RiskLimits,
        *,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self.limits = limits
        self._kill_switch = KillSwitch()
        self._time = time_source or time.time
        self._positions: MutableMapping[str, float] = {}
        self._last_notional: MutableMapping[str, float] = {}
        self._submissions: deque[float] = deque()
        self._avg_prices: MutableMapping[str, float] = {}
        self._realized_pnl = 0.0
        self._soft_limit_breached = False
        self._soft_limit_reason = ""

    def _canonical_symbol(self, symbol: str) -> str:
        return normalize_symbol(symbol)

    def _check_rate_limit(self, now: float) -> None:
        if self.limits.max_orders_per_interval <= 0:
            return
        window = max(self.limits.interval_seconds, 0.0)
        while self._submissions and now - self._submissions[0] > window:
            self._submissions.popleft()
        if len(self._submissions) >= self.limits.max_orders_per_interval:
            raise OrderRateExceeded(
                f"Order throttle exceeded: {len(self._submissions)} submissions in {window}s",
            )

    def _validate_inputs(self, qty: float, price: float) -> None:
        if qty < 0:
            raise ValueError("quantity must be non-negative")
        if price <= 0:
            raise ValueError("price must be positive")

    def validate_order(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Validate an order request without mutating the position book."""

        self._kill_switch.guard()
        self._validate_inputs(qty, price)
        self._check_loss_limits()
        canonical_symbol = self._canonical_symbol(symbol)
        now = self._time()
        self._check_rate_limit(now)

        side_sign = 1.0 if side.lower() == "buy" else -1.0
        current_position = float(self._positions.get(canonical_symbol, 0.0))
        new_position = current_position + side_sign * qty

        if abs(new_position) > self.limits.max_position:
            raise LimitViolation(
                f"Position cap exceeded for {canonical_symbol}: {new_position} > {self.limits.max_position}",
            )

        projected_notional = abs(new_position * price)
        if projected_notional > self.limits.max_notional:
            raise LimitViolation(
                f"Notional cap exceeded for {canonical_symbol}: {projected_notional} > {self.limits.max_notional}",
            )

        self._submissions.append(now)

    @property
    def kill_switch(self) -> KillSwitch:
        """Expose the kill-switch handle."""

        return self._kill_switch

    def register_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Update exposure after a confirmed fill."""

        self._validate_inputs(qty, price)
        canonical_symbol = self._canonical_symbol(symbol)
        side_sign = 1.0 if side.lower() == "buy" else -1.0
        trade_qty = side_sign * qty

        position = float(self._positions.get(canonical_symbol, 0.0))
        avg_price = float(self._avg_prices.get(canonical_symbol, price))
        realized = 0.0

        if position > 0 and trade_qty < 0:
            closing = min(position, -trade_qty)
            realized += closing * (price - avg_price)
            position -= closing
            trade_qty += closing
        elif position < 0 and trade_qty > 0:
            closing = min(-position, trade_qty)
            realized += closing * (avg_price - price)
            position += closing
            trade_qty -= closing

        if trade_qty > 0:
            # Increasing or flipping to a long position
            new_position = position + trade_qty
            if new_position != 0:
                existing = position if position > 0 else 0.0
                total = existing + trade_qty
                blended = price if existing == 0 else (avg_price * existing + price * trade_qty) / total
                self._avg_prices[canonical_symbol] = blended
            else:
                self._avg_prices.pop(canonical_symbol, None)
            position = new_position
        elif trade_qty < 0:
            # Increasing or flipping to a short position
            new_position = position + trade_qty
            if new_position != 0:
                existing = -position if position < 0 else 0.0
                total = existing + (-trade_qty)
                blended = price if existing == 0 else (avg_price * existing + price * (-trade_qty)) / total
                self._avg_prices[canonical_symbol] = blended
            else:
                self._avg_prices.pop(canonical_symbol, None)
            position = new_position
        else:
            if position == 0:
                self._avg_prices.pop(canonical_symbol, None)

        if abs(position) < 1e-12:
            position = 0.0
            self._positions.pop(canonical_symbol, None)
            self._last_notional.pop(canonical_symbol, None)
        else:
            self._positions[canonical_symbol] = position
            self._last_notional[canonical_symbol] = abs(position * price)

        self._realized_pnl += realized
        self._update_loss_state()

    def current_position(self, symbol: str) -> float:
        canonical_symbol = self._canonical_symbol(symbol)
        return float(self._positions.get(canonical_symbol, 0.0))

    def current_notional(self, symbol: str) -> float:
        canonical_symbol = self._canonical_symbol(symbol)
        return float(self._last_notional.get(canonical_symbol, 0.0))

    @property
    def realized_pnl(self) -> float:
        """Return the cumulative realised PnL tracked by the manager."""

        return float(self._realized_pnl)

    def reset_realized_pnl(self) -> None:
        """Reset PnL and soft limit state (e.g. start of trading session).

        The kill-switch state is intentionally preserved; callers that need a
        fresh trading session must reset :attr:`kill_switch` explicitly.
        """

        self._realized_pnl = 0.0
        self._soft_limit_breached = False
        self._soft_limit_reason = ""

    # ------------------------------------------------------------------
    # Loss-limit helpers
    def _update_loss_state(self) -> None:
        max_loss = float(self.limits.max_daily_loss)
        if max_loss <= 0 or math.isinf(max_loss):
            return

        realized = self._realized_pnl
        threshold = -abs(max_loss)
        ratio = float(self.limits.soft_loss_ratio)
        soft_threshold = threshold * ratio

        if realized <= threshold:
            if not self._kill_switch.is_triggered():
                self._kill_switch.trigger(
                    f"Max loss breached ({realized:.2f} <= {-abs(max_loss):.2f})"
                )
            self._soft_limit_breached = True
            self._soft_limit_reason = (
                f"Max loss limit reached: realised PnL {realized:.2f} <= {-abs(max_loss):.2f}"
            )
            return

        if ratio > 0 and realized <= soft_threshold:
            self._soft_limit_breached = True
            self._soft_limit_reason = (
                f"Soft loss limit reached: realised PnL {realized:.2f} <= {soft_threshold:.2f}"
            )
        else:
            self._soft_limit_breached = False
            self._soft_limit_reason = ""

    def _check_loss_limits(self) -> None:
        if not self._soft_limit_breached:
            return
        raise LimitViolation(self._soft_limit_reason or "Loss limits breached")


class IdempotentRetryExecutor:
    """Retry wrapper that guarantees idempotency via explicit keys."""

    def __init__(self, *, backoff: Callable[[int], float] | None = None) -> None:
        self._results: Dict[str, object] = {}
        self._backoff = backoff

    def run(
        self,
        key: str,
        func: Callable[[int], object],
        retries: int = 3,
        retry_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> object:
        if key in self._results:
            return self._results[key]

        attempt = 0
        last_error: Exception | None = None
        while attempt < retries:
            attempt += 1
            try:
                result = func(attempt)
                self._results[key] = result
                return result
            except retry_exceptions as exc:  # type: ignore[misc]
                last_error = exc
                if attempt >= retries:
                    raise
                if self._backoff is not None:
                    delay = max(0.0, float(self._backoff(attempt)))
                    if delay:
                        time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("IdempotentRetryExecutor terminated without executing the callable")


class DefaultPortfolioRiskAnalyzer(PortfolioRiskAnalyzer):
    """Portfolio risk analyzer compatible with :func:`portfolio_heat`."""

    def heat(self, positions: Iterable[Mapping[str, float]]) -> float:
        total = 0.0
        for pos in positions:
            qty = float(pos.get("qty", 0.0))
            price = float(pos.get("price", 0.0))
            risk_weight = float(pos.get("risk_weight", 1.0))
            side = pos.get("side", "long")
            direction = 1.0 if side == "long" else -1.0
            total += abs(qty * price * risk_weight * direction)
        return float(total)


def portfolio_heat(positions: Iterable[Mapping[str, float]]) -> float:
    """Compute aggregate risk heat with directionality and weights."""

    analyzer = DefaultPortfolioRiskAnalyzer()
    return analyzer.heat(positions)


__all__ = [
    "RiskError",
    "LimitViolation",
    "OrderRateExceeded",
    "RiskLimits",
    "KillSwitch",
    "RiskManager",
    "DefaultPortfolioRiskAnalyzer",
    "IdempotentRetryExecutor",
    "portfolio_heat",
]
