# SPDX-License-Identifier: MIT
"""Sandbox-friendly broker connectors for major exchanges."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping

from domain import Order

from .normalization import NormalizationError, SymbolNormalizer, SymbolSpecification


class OrderError(RuntimeError):
    """Raised when a connector fails to process an order."""


class ExecutionConnector:
    """Minimal connector interface shared across exchanges."""

    def __init__(self, *, sandbox: bool = True) -> None:
        self.sandbox = sandbox

    def connect(self, credentials: Mapping[str, str] | None = None) -> None:
        """Connect to the venue. Sandbox connectors are no-ops."""

    def disconnect(self) -> None:
        """Disconnect from the venue."""

    def place_order(self, order: Order) -> Order:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def fetch_order(self, order_id: str) -> Order:
        raise NotImplementedError

    def open_orders(self) -> Iterable[Order]:
        raise NotImplementedError

    def get_positions(self) -> List[dict]:
        raise NotImplementedError


class SimulatedExchangeConnector(ExecutionConnector):
    """Base class providing deterministic sandbox behaviour."""

    def __init__(
        self,
        *,
        sandbox: bool = True,
        symbol_map: Mapping[str, str] | None = None,
        specifications: Mapping[str, SymbolSpecification] | None = None,
    ) -> None:
        super().__init__(sandbox=sandbox)
        self.normalizer = SymbolNormalizer(symbol_map, specifications)
        self._orders: Dict[str, Order] = {}
        self._next_id = 1

    def _generate_id(self) -> str:
        order_id = f"{self.__class__.__name__}-{self._next_id:08d}"
        self._next_id += 1
        return order_id

    def place_order(self, order: Order) -> Order:
        quantity = self.normalizer.round_quantity(order.symbol, order.quantity)
        price = order.price
        if price is not None:
            price = self.normalizer.round_price(order.symbol, price)
        try:
            self.normalizer.validate(order.symbol, quantity, price)
        except NormalizationError as exc:  # pragma: no cover - defensive guard
            raise OrderError(str(exc)) from exc
        normalized_order = replace(order, quantity=quantity, price=price)
        order_id = self._generate_id()
        normalized_order.mark_submitted(order_id)
        self._orders[order_id] = normalized_order
        return normalized_order

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        order.cancel()
        return True

    def fetch_order(self, order_id: str) -> Order:
        if order_id not in self._orders:
            raise OrderError(f"Unknown order_id: {order_id}")
        return self._orders[order_id]

    def open_orders(self) -> Iterable[Order]:
        return [order for order in self._orders.values() if order.is_active]

    def get_positions(self) -> List[dict]:
        return []

    def apply_fill(self, order_id: str, quantity: float, price: float) -> Order:
        order = self.fetch_order(order_id)
        order.record_fill(quantity, price)
        return order


class BinanceConnector(SimulatedExchangeConnector):
    def __init__(self, *, sandbox: bool = True) -> None:
        specs = {
            "BTCUSDT": SymbolSpecification("BTCUSDT", min_qty=0.0001, min_notional=10, step_size=0.0001, tick_size=0.1),
            "ETHUSDT": SymbolSpecification("ETHUSDT", min_qty=0.001, min_notional=5, step_size=0.001, tick_size=0.01),
        }
        super().__init__(sandbox=sandbox, symbol_map={"BTCUSDT": "BTCUSDT"}, specifications=specs)


class BybitConnector(SimulatedExchangeConnector):
    def __init__(self, *, sandbox: bool = True) -> None:
        specs = {
            "BTCUSDT": SymbolSpecification("BTCUSDT", min_qty=0.001, min_notional=5, step_size=0.001, tick_size=0.5),
        }
        super().__init__(sandbox=sandbox, symbol_map={"BTCUSDT": "BTCUSDT"}, specifications=specs)


class KrakenConnector(SimulatedExchangeConnector):
    def __init__(self, *, sandbox: bool = True) -> None:
        symbol_map = {"BTCUSD": "XBTUSD"}
        specs = {
            "XBTUSD": SymbolSpecification("XBTUSD", min_qty=0.0001, min_notional=5, step_size=0.0001, tick_size=0.5),
        }
        super().__init__(sandbox=sandbox, symbol_map=symbol_map, specifications=specs)


class CoinbaseConnector(SimulatedExchangeConnector):
    def __init__(self, *, sandbox: bool = True) -> None:
        symbol_map = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD"}
        specs = {
            "BTC-USD": SymbolSpecification("BTC-USD", min_qty=0.0001, min_notional=10, step_size=0.0001, tick_size=0.01),
            "ETH-USD": SymbolSpecification("ETH-USD", min_qty=0.001, min_notional=5, step_size=0.001, tick_size=0.01),
        }
        super().__init__(sandbox=sandbox, symbol_map=symbol_map, specifications=specs)


__all__ = [
    "OrderError",
    "ExecutionConnector",
    "SimulatedExchangeConnector",
    "BinanceConnector",
    "BybitConnector",
    "KrakenConnector",
    "CoinbaseConnector",
]
