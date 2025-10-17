# SPDX-License-Identifier: MIT
"""Utilities for exchange-specific symbol and quantity normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(slots=True, frozen=True)
class SymbolSpecification:
    """Trading constraints for a specific symbol on an exchange."""

    symbol: str
    min_qty: float = 0.0
    min_notional: float = 0.0
    step_size: float = 0.0
    tick_size: float = 0.0


class NormalizationError(ValueError):
    """Raised when a quantity or price violates exchange constraints."""


class SymbolNormalizer:
    """Normalize symbols and enforce venue-specific constraints."""

    def __init__(
        self,
        symbol_map: Mapping[str, str] | None = None,
        specifications: Mapping[str, SymbolSpecification] | None = None,
    ) -> None:
        self._symbol_map: Dict[str, str] = {
            self._canonical(k): v for k, v in (symbol_map or {}).items()
        }
        self._specs: Dict[str, SymbolSpecification] = {
            self._canonical(spec.symbol): spec
            for spec in (specifications or {}).values()
        }

    @staticmethod
    def _canonical(symbol: str) -> str:
        return symbol.replace("-", "").replace("_", "").upper()

    def exchange_symbol(self, symbol: str) -> str:
        canonical = self._canonical(symbol)
        return self._symbol_map.get(canonical, canonical)

    def specification(self, symbol: str) -> SymbolSpecification | None:
        canonical = self._canonical(symbol)
        if canonical in self._specs:
            return self._specs[canonical]
        exchange_symbol = self._symbol_map.get(canonical)
        if exchange_symbol is not None:
            canonical_exchange = self._canonical(exchange_symbol)
            return self._specs.get(canonical_exchange)
        return None

    @staticmethod
    def _round(value: float, step: float) -> float:
        if step <= 0:
            return value
        steps = round(value / step)
        return round(steps * step, 12)

    def round_quantity(self, symbol: str, quantity: float) -> float:
        spec = self.specification(symbol)
        if spec is None:
            return quantity
        return self._round(quantity, spec.step_size)

    def round_price(self, symbol: str, price: float) -> float:
        spec = self.specification(symbol)
        if spec is None:
            return price
        return self._round(price, spec.tick_size)

    def validate(
        self, symbol: str, quantity: float, price: float | None = None
    ) -> None:
        spec = self.specification(symbol)
        if spec is None:
            return
        if quantity < spec.min_qty:
            raise NormalizationError(
                f"Quantity {quantity} below minimum {spec.min_qty} for {symbol}"
            )
        if price is not None and spec.min_notional:
            notional = quantity * price
            if notional < spec.min_notional:
                raise NormalizationError(
                    f"Notional {notional} below minimum {spec.min_notional} for {symbol}"
                )


__all__ = [
    "SymbolSpecification",
    "NormalizationError",
    "SymbolNormalizer",
]
