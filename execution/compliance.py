"""Pre-trade compliance checks against exchange minimums."""

from __future__ import annotations

from dataclasses import dataclass

from .normalization import NormalizationError, SymbolNormalizer

__all__ = [
    "ComplianceViolation",
    "ComplianceReport",
    "ComplianceMonitor",
]


class ComplianceViolation(NormalizationError):
    """Raised when an order violates venue-level minimums."""


@dataclass(slots=True, frozen=True)
class ComplianceReport:
    """Outcome of a pre-trade compliance check."""

    symbol: str
    requested_quantity: float
    requested_price: float | None
    normalized_quantity: float
    normalized_price: float | None
    violations: tuple[str, ...]
    blocked: bool

    def is_clean(self) -> bool:
        return not self.violations


class ComplianceMonitor:
    """Validate orders against lot, tick, and notional minimums before routing."""

    def __init__(self, normalizer: SymbolNormalizer, *, strict: bool = True, auto_round: bool = True) -> None:
        self._normalizer = normalizer
        self._strict = strict
        self._auto_round = auto_round

    def check(self, symbol: str, quantity: float, price: float | None = None) -> ComplianceReport:
        normalized_qty = self._normalizer.round_quantity(symbol, quantity) if self._auto_round else quantity
        normalized_price = (
            None if price is None else self._normalizer.round_price(symbol, price)
        ) if self._auto_round else price

        violations: list[str] = []
        try:
            self._normalizer.validate(symbol, normalized_qty, normalized_price)
        except NormalizationError as exc:
            violations.append(str(exc))
            if self._strict:
                raise ComplianceViolation(str(exc)) from exc

        blocked = bool(violations) and self._strict
        return ComplianceReport(
            symbol=symbol,
            requested_quantity=float(quantity),
            requested_price=None if price is None else float(price),
            normalized_quantity=float(normalized_qty),
            normalized_price=None if normalized_price is None else float(normalized_price),
            violations=tuple(violations),
            blocked=blocked,
        )
