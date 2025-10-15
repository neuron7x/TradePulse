# SPDX-License-Identifier: MIT
"""Execution risk controls with kill-switch governance and telemetry hooks.

This module houses the reference :class:`RiskManager` used by the live trading
runner (see ``docs/runbook_live_trading.md``) and the risk, signals, and
observability blueprint in ``docs/risk_ml_observability.md``. It enforces
position/notional limits, order-rate throttles, and a kill-switch escalation
mechanism aligned with the governance expectations formalised in
``docs/documentation_governance.md`` and ``docs/monitoring.md``.

The implementation depends on catalog normalisation utilities, execution audit
logging, and metrics collectors to ensure every decision is observable and
attributable—an explicit requirement in ``docs/quality_gates.md``.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Protocol

from core.data.catalog import normalize_symbol
from core.utils.logging import get_logger
from core.utils.metrics import get_metrics_collector
from interfaces.execution import PortfolioRiskAnalyzer, RiskController

from .audit import ExecutionAuditLogger, get_execution_audit_logger


class RiskError(RuntimeError):
    """Base exception for risk-control violations."""


class LimitViolation(RiskError):
    """Raised when position or notional limits would be breached."""


class OrderRateExceeded(RiskError):
    """Raised when the order rate throttle blocks a submission."""


@dataclass(slots=True)
class RiskLimits:
    """Risk guardrails that must be enforced prior to execution.

    Attributes:
        max_notional: Absolute notional cap per instrument.
        max_position: Signed position cap.
        max_orders_per_interval: Order submissions allowed within ``interval_seconds``.
        interval_seconds: Rolling window length for the throttle.
        kill_switch_limit_multiplier: Severity multiplier that instantly trips the
            kill-switch when exceeded.
        kill_switch_violation_threshold: Consecutive limit breaches that trigger
            the kill-switch.
        kill_switch_rate_limit_threshold: Consecutive throttle breaches that
            trigger the kill-switch.
    """

    max_notional: float = float("inf")
    max_position: float = float("inf")
    max_orders_per_interval: int = 60
    interval_seconds: float = 1.0
    kill_switch_limit_multiplier: float = 1.5
    kill_switch_violation_threshold: int = 3
    kill_switch_rate_limit_threshold: int = 3

    def __post_init__(self) -> None:
        if self.max_orders_per_interval < 0:
            self.max_orders_per_interval = 0
        if self.interval_seconds < 0:
            self.interval_seconds = 0.0
        if self.kill_switch_limit_multiplier < 1.0:
            self.kill_switch_limit_multiplier = 1.0
        if self.kill_switch_violation_threshold < 1:
            self.kill_switch_violation_threshold = 1
        if self.kill_switch_rate_limit_threshold < 1:
            self.kill_switch_rate_limit_threshold = 1


class KillSwitchStateStore(Protocol):
    """Persistence backend for kill-switch state."""

    def load(self) -> tuple[bool, str] | None:
        """Return the last persisted state, if any."""

    def save(self, engaged: bool, reason: str) -> None:
        """Persist the supplied state atomically."""


class SQLiteKillSwitchStateStore(KillSwitchStateStore):
    """SQLite-backed store used to persist kill-switch state across restarts."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._initialise()

    def _initialise(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    engaged INTEGER NOT NULL CHECK (engaged IN (0, 1)),
                    reason TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load(self) -> tuple[bool, str] | None:
        with sqlite3.connect(self._path) as connection:
            cursor = connection.execute(
                "SELECT engaged, reason FROM kill_switch_state WHERE id = 1"
            )
            row = cursor.fetchone()
        if row is None:
            return None
        engaged, reason = row
        return bool(engaged), reason or ""

    def save(self, engaged: bool, reason: str) -> None:
        payload_reason = reason or ""
        with self._lock:
            with sqlite3.connect(self._path) as connection:
                connection.execute(
                    """
                    INSERT INTO kill_switch_state (id, engaged, reason, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        engaged = excluded.engaged,
                        reason = excluded.reason,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (1, int(bool(engaged)), payload_reason),
                )


class KillSwitch:
    """Global kill-switch toggled on critical failures with optional persistence.

    The kill-switch mirrors the operational blueprint in
    ``docs/admin_remote_control.md`` and is surfaced via CLI and observability
    tooling for rapid operator response. When supplied with a
    :class:`KillSwitchStateStore` it reloads the persisted state during
    initialisation to preserve operator intent across restarts.
    """

    def __init__(self, store: KillSwitchStateStore | None = None) -> None:
        self._store = store
        self._triggered = False
        self._reason = ""
        if self._store is not None:
            persisted = self._store.load()
            if persisted is not None:
                engaged, reason = persisted
                self._triggered = bool(engaged)
                self._reason = reason

    def trigger(self, reason: str) -> None:
        """Engage the kill-switch with an explanatory ``reason``."""

        self._triggered = True
        self._reason = reason
        if self._store is not None:
            self._store.save(True, reason)

    def reset(self) -> None:
        """Clear the kill-switch state."""

        self._triggered = False
        self._reason = ""
        if self._store is not None:
            self._store.save(False, "")

    @property
    def reason(self) -> str:
        """Return the human-readable explanation for the last trigger."""

        return self._reason

    def is_triggered(self) -> bool:
        """Indicate whether the kill-switch is currently engaged."""

        return self._triggered

    def guard(self) -> None:
        """Raise :class:`RiskError` if the kill-switch is active."""

        if self._triggered:
            raise RiskError(f"Kill-switch engaged: {self._reason or 'unspecified reason'}")


class RiskManager(RiskController):
    """Apply notional/position caps and order throttling.

    The manager coordinates with :mod:`execution.audit`, records metrics for the
    observability pipeline in ``docs/monitoring.md`` and enforces the governance
    rules codified in ``docs/documentation_governance.md`` before orders reach
    the venue adapters.
    """

    def __init__(
        self,
        limits: RiskLimits,
        *,
        time_source: Callable[[], float] | None = None,
        audit_logger: ExecutionAuditLogger | None = None,
        kill_switch_store: KillSwitchStateStore | None = None,
    ) -> None:
        self.limits = limits
        self._kill_switch = KillSwitch(store=kill_switch_store)
        self._time = time_source or time.time
        self._positions: MutableMapping[str, float] = {}
        self._last_notional: MutableMapping[str, float] = {}
        self._submissions: deque[float] = deque()
        self._logger = get_logger(__name__)
        self._metrics = get_metrics_collector()
        self._audit = audit_logger or get_execution_audit_logger()
        self._limit_violation_streak = 0
        self._throttle_violation_streak = 0

    def _canonical_symbol(self, symbol: str) -> str:
        return normalize_symbol(symbol)

    def _check_rate_limit(self, symbol: str, now: float) -> None:
        if self.limits.max_orders_per_interval <= 0:
            return
        window = max(self.limits.interval_seconds, 0.0)
        while self._submissions and now - self._submissions[0] > window:
            self._submissions.popleft()
        if len(self._submissions) >= self.limits.max_orders_per_interval:
            self._throttle_violation_streak += 1
            reason = (
                f"Order throttle exceeded: {len(self._submissions)} submissions in {window}s"
            )
            if self._throttle_violation_streak >= self.limits.kill_switch_rate_limit_threshold:
                self._trigger_kill_switch(
                    reason,
                    symbol=symbol,
                    violation_type="rate_limit",
                )
            raise OrderRateExceeded(reason)
        self._throttle_violation_streak = 0

    def _validate_inputs(self, qty: float, price: float) -> None:
        if qty < 0:
            raise ValueError("quantity must be non-negative")
        if price <= 0:
            raise ValueError("price must be positive")

    def _record_risk_audit(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str,
        reason: str | None = None,
        violation_type: str | None = None,
    ) -> None:
        payload = {
            "event": "risk_validation",
            "status": status,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "reason": reason,
            "violation_type": violation_type,
            "kill_switch_engaged": self._kill_switch.is_triggered(),
            "consecutive_limit_violations": self._limit_violation_streak,
            "consecutive_rate_limit_violations": self._throttle_violation_streak,
        }
        self._audit.emit(payload)

    def _trigger_kill_switch(
        self,
        reason: str,
        *,
        symbol: str | None = None,
        violation_type: str | None = None,
    ) -> None:
        if self._kill_switch.is_triggered():
            return
        self._kill_switch.trigger(reason)
        self._logger.critical(
            "Kill switch triggered",
            reason=reason,
            symbol=symbol,
            violation_type=violation_type,
            consecutive_limit_violations=self._limit_violation_streak,
            consecutive_rate_limit_violations=self._throttle_violation_streak,
        )
        self._metrics.record_kill_switch_trigger(reason)
        self._audit.emit(
            {
                "event": "kill_switch_triggered",
                "reason": reason,
                "symbol": symbol,
                "violation_type": violation_type,
                "consecutive_limit_violations": self._limit_violation_streak,
                "consecutive_rate_limit_violations": self._throttle_violation_streak,
            }
        )

    def validate_order(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Apply risk checks before admitting an order to the execution stack.

        Args:
            symbol: External instrument identifier; normalised via
                :func:`core.data.catalog.normalize_symbol`.
            side: Case-insensitive trade direction (``"buy"`` or ``"sell"``).
            qty: Order quantity expressed in base units. Must be non-negative.
            price: Reference price used for notional calculations. Must be
                strictly positive.

        Raises:
            ValueError: If ``qty`` or ``price`` fall outside allowable ranges.
            RiskError: When the kill-switch is already engaged.
            OrderRateExceeded: If the rate limiter blocks the submission.
            LimitViolation: When position or notional caps would be breached.

        Examples:
            >>> limits = RiskLimits(max_notional=10_000, max_position=5)
            >>> manager = RiskManager(limits)
            >>> manager.validate_order("BTC-USD", "buy", 1, 25_000.0)
            Traceback (most recent call last):
            ...
            LimitViolation: Notional cap exceeded for BTC-USD: 25000.0 > 10000.0

        Notes:
            - Successful validation appends the submission timestamp for rate-limit
              accounting and emits telemetry via :mod:`execution.audit`.
            - Consecutive limit or throttle violations trigger the kill-switch when
              thresholds in :class:`RiskLimits` are met, matching the operational
              response plan in ``docs/admin_remote_control.md``.
        """

        self._validate_inputs(qty, price)
        canonical_symbol = self._canonical_symbol(symbol)
        try:
            self._kill_switch.guard()
        except RiskError as exc:
            self._metrics.record_risk_validation(canonical_symbol, "kill_switch_blocked")
            self._record_risk_audit(
                symbol=canonical_symbol,
                side=side.lower(),
                quantity=float(qty),
                price=float(price),
                status="blocked",
                reason=str(exc),
                violation_type="kill_switch",
            )
            raise
        now = self._time()
        try:
            self._check_rate_limit(canonical_symbol, now)
        except OrderRateExceeded as exc:
            reason = str(exc)
            self._metrics.record_risk_validation(canonical_symbol, "rate_limited")
            self._record_risk_audit(
                symbol=canonical_symbol,
                side=side.lower(),
                quantity=float(qty),
                price=float(price),
                status="rejected",
                reason=reason,
                violation_type="rate_limit",
            )
            raise

        side_sign = 1.0 if side.lower() == "buy" else -1.0
        current_position = float(self._positions.get(canonical_symbol, 0.0))
        new_position = current_position + side_sign * qty

        if abs(new_position) > self.limits.max_position:
            reason = (
                f"Position cap exceeded for {canonical_symbol}: "
                f"{new_position} > {self.limits.max_position}"
            )
            self._limit_violation_streak += 1
            severe = abs(new_position) > (
                self.limits.max_position * self.limits.kill_switch_limit_multiplier
            )
            if severe or (
                self._limit_violation_streak
                >= self.limits.kill_switch_violation_threshold
            ):
                self._trigger_kill_switch(
                    reason,
                    symbol=canonical_symbol,
                    violation_type="position_limit",
                )
            self._metrics.record_risk_validation(canonical_symbol, "position_limit")
            self._record_risk_audit(
                symbol=canonical_symbol,
                side=side.lower(),
                quantity=float(qty),
                price=float(price),
                status="rejected",
                reason=reason,
                violation_type="position_limit",
            )
            raise LimitViolation(reason)

        projected_notional = abs(new_position * price)
        if projected_notional > self.limits.max_notional:
            reason = (
                f"Notional cap exceeded for {canonical_symbol}: "
                f"{projected_notional} > {self.limits.max_notional}"
            )
            self._limit_violation_streak += 1
            severe = projected_notional > (
                self.limits.max_notional * self.limits.kill_switch_limit_multiplier
            )
            if severe or (
                self._limit_violation_streak
                >= self.limits.kill_switch_violation_threshold
            ):
                self._trigger_kill_switch(
                    reason,
                    symbol=canonical_symbol,
                    violation_type="notional_limit",
                )
            self._metrics.record_risk_validation(canonical_symbol, "notional_limit")
            self._record_risk_audit(
                symbol=canonical_symbol,
                side=side.lower(),
                quantity=float(qty),
                price=float(price),
                status="rejected",
                reason=reason,
                violation_type="notional_limit",
            )
            raise LimitViolation(reason)

        if self.limits.max_orders_per_interval > 0:
            self._submissions.append(now)
        else:
            self._submissions.clear()
        self._limit_violation_streak = 0
        self._metrics.record_risk_validation(canonical_symbol, "passed")
        self._record_risk_audit(
            symbol=canonical_symbol,
            side=side.lower(),
            quantity=float(qty),
            price=float(price),
            status="passed",
            reason=None,
            violation_type=None,
        )

    @property
    def kill_switch(self) -> KillSwitch:
        """Expose the kill-switch handle."""

        return self._kill_switch

    def register_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Update exposure after a confirmed fill.

        Args:
            symbol: External instrument identifier; normalised via
                :func:`core.data.catalog.normalize_symbol`.
            side: Executed direction (``"buy"`` or ``"sell"``).
            qty: Fill quantity in base units.
            price: Fill price used for notional tracking.

        Notes:
            Exposure updates feed portfolio analytics described in
            ``docs/risk_ml_observability.md``.
        """

        self._validate_inputs(qty, price)
        canonical_symbol = self._canonical_symbol(symbol)
        side_sign = 1.0 if side.lower() == "buy" else -1.0
        position = float(self._positions.get(canonical_symbol, 0.0)) + side_sign * qty
        self._positions[canonical_symbol] = position
        self._last_notional[canonical_symbol] = abs(position * price)

    def current_position(self, symbol: str) -> float:
        """Return the signed position for ``symbol``."""

        canonical_symbol = self._canonical_symbol(symbol)
        return float(self._positions.get(canonical_symbol, 0.0))

    def current_notional(self, symbol: str) -> float:
        """Return the absolute notional exposure for ``symbol``."""

        canonical_symbol = self._canonical_symbol(symbol)
        return float(self._last_notional.get(canonical_symbol, 0.0))


class IdempotentRetryExecutor:
    """Retry wrapper that guarantees idempotency via explicit keys.

    The executor is designed for side-effect-free RPCs such as venue state
    queries and follows the retry governance guidance in ``docs/execution.md``.
    """

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
        """Execute ``func`` with retries while caching the first success.

        Args:
            key: Cache key ensuring idempotent semantics.
            func: Callable receiving the attempt count (starting at ``1``).
            retries: Maximum number of attempts.
            retry_exceptions: Tuple of exception types that should trigger retry
                behaviour.

        Returns:
            object: The result of ``func`` from the first successful attempt.

        Raises:
            Exception: Re-raises the last exception when retries are exhausted.

        Examples:
            >>> executor = IdempotentRetryExecutor()
            >>> executor.run("ping", lambda attempt: attempt)
            1
        """

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
    """Portfolio risk analyzer compatible with :func:`portfolio_heat`.

    The analyzer mirrors the directional heat aggregation described in
    ``docs/risk_ml_observability.md`` and provides a lightweight default for
    CLI tooling and notebooks.
    """

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
    """Compute aggregate risk heat with directionality and weights.

    Args:
        positions: Iterable of dictionaries containing ``qty``, ``price``,
            ``risk_weight``, and ``side`` keys.

    Returns:
        float: Absolute risk heat used for monitoring dashboards in
        ``docs/monitoring.md``.
    """

    analyzer = DefaultPortfolioRiskAnalyzer()
    return analyzer.heat(positions)


__all__ = [
    "RiskError",
    "LimitViolation",
    "OrderRateExceeded",
    "RiskLimits",
    "KillSwitchStateStore",
    "SQLiteKillSwitchStateStore",
    "KillSwitch",
    "RiskManager",
    "DefaultPortfolioRiskAnalyzer",
    "IdempotentRetryExecutor",
    "portfolio_heat",
]
