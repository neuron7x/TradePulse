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

    def __init__(self, message: str, *, report: "ComplianceReport" | None = None) -> None:
        super().__init__(message)
        self.report = report


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

    def to_dict(self) -> dict:
        """Serialize the report to a plain dictionary."""

        return {
            "symbol": self.symbol,
            "requested_quantity": self.requested_quantity,
            "requested_price": self.requested_price,
            "normalized_quantity": self.normalized_quantity,
            "normalized_price": self.normalized_price,
            "violations": list(self.violations),
            "blocked": self.blocked,
        }


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
        violation_exc: NormalizationError | None = None
        try:
            self._normalizer.validate(symbol, normalized_qty, normalized_price)
        except NormalizationError as exc:
            violations.append(str(exc))
            violation_exc = exc

        blocked = bool(violations) and self._strict
        report = ComplianceReport(
            symbol=symbol,
            requested_quantity=float(quantity),
            requested_price=None if price is None else float(price),
            normalized_quantity=float(normalized_qty),
            normalized_price=None if normalized_price is None else float(normalized_price),
            violations=tuple(violations),
            blocked=blocked,
        )
        if violation_exc is not None and self._strict:
            raise ComplianceViolation(str(violation_exc), report=report) from violation_exc
        return report
