# SPDX-License-Identifier: MIT
"""Deterministic in-memory limit-order-book simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from typing import Callable, Deque, Dict, Iterator, List, MutableMapping, Optional

from domain import Order, OrderSide, OrderType

__all__ = [
    "CancelEvent",
    "FillEvent",
    "LimitOrderBookSimulator",
]


@dataclass(slots=True)
class FillEvent:
    """Fill notification emitted by the simulator."""

    order_id: str
    counterparty_id: str
    quantity: float
    price: float
    liquidity: str
    timestamp: datetime


@dataclass(slots=True)
class CancelEvent:
    """Cancellation notification emitted by the simulator."""

    order_id: str
    reason: str | None
    timestamp: datetime


class _BookOrder:
    """Internal representation of a resting order."""

    __slots__ = ("order", "remaining")

    def __init__(self, order: Order) -> None:
        self.order = order
        self.remaining = order.remaining_quantity

    def refresh(self) -> None:
        self.remaining = self.order.remaining_quantity


class _PriceLevel:
    """FIFO queue for orders resting at the same price."""

    __slots__ = ("price", "_orders")

    def __init__(self, price: float) -> None:
        self.price = price
        self._orders: Deque[_BookOrder] = deque()

    def append(self, book_order: _BookOrder) -> None:
        self._orders.append(book_order)

    def peek(self) -> Optional[_BookOrder]:
        self._cull()
        return self._orders[0] if self._orders else None

    def popleft(self) -> Optional[_BookOrder]:
        self._cull()
        return self._orders.popleft() if self._orders else None

    def remove(self, order_id: str) -> bool:
        removed = False
        kept: Deque[_BookOrder] = deque()
        while self._orders:
            candidate = self._orders.popleft()
            if candidate.order.order_id == order_id:
                removed = True
                continue
            kept.append(candidate)
        self._orders = kept
        return removed

    def __bool__(self) -> bool:  # pragma: no cover - defensive guard
        self._cull()
        return bool(self._orders)

    def _cull(self) -> None:
        while self._orders and self._orders[0].order.remaining_quantity <= 1e-12:
            self._orders.popleft()


class _SideBook:
    """Price-aggregated view of one side of the book."""

    __slots__ = ("side", "_levels", "_sorted_prices")

    def __init__(self, side: OrderSide) -> None:
        self.side = side
        self._levels: MutableMapping[float, _PriceLevel] = {}
        self._sorted_prices: List[float] = []

    def add(self, price: float, order: _BookOrder) -> None:
        level = self._levels.get(price)
        if level is None:
            level = _PriceLevel(price)
            self._levels[price] = level
            self._sorted_prices.append(price)
            self._sorted_prices.sort(reverse=self.side == OrderSide.BUY)
        level.append(order)

    def best(self) -> Optional[_PriceLevel]:
        self._drop_empty()
        if not self._sorted_prices:
            return None
        price = self._sorted_prices[0]
        return self._levels[price]

    def remove(self, price: float, order_id: str) -> bool:
        level = self._levels.get(price)
        if level is None:
            return False
        removed = level.remove(order_id)
        if removed and not level.peek():
            self._levels.pop(price, None)
            self._sorted_prices = [p for p in self._sorted_prices if p != price]
        return removed

    def pop_level_if_empty(self, price: float) -> None:
        level = self._levels.get(price)
        if level and not level.peek():
            self._levels.pop(price, None)
            self._sorted_prices = [p for p in self._sorted_prices if p != price]

    def iter_orders(self) -> Iterator[Order]:
        self._drop_empty()
        for price in self._sorted_prices:
            level = self._levels[price]
            for book_order in level._orders:
                if book_order.order.is_active:
                    yield book_order.order

    def _drop_empty(self) -> None:
        retained: List[float] = []
        for price in self._sorted_prices:
            level = self._levels.get(price)
            if level is None:
                continue
            if level.peek() is None:
                self._levels.pop(price, None)
                continue
            retained.append(price)
        self._sorted_prices = retained


class _SymbolBook:
    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        self.bids = _SideBook(OrderSide.BUY)
        self.asks = _SideBook(OrderSide.SELL)


class LimitOrderBookSimulator:
    """Minimal deterministic in-memory exchange simulator."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        start = 1 if seed is None else max(int(seed), 1)
        self._id_counter = count(start)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._books: Dict[str, _SymbolBook] = {}
        self._order_index: Dict[str, tuple[str, OrderSide, float]] = {}
        self._orders: Dict[str, Order] = {}
        self._events: Deque[FillEvent | CancelEvent] = deque()

    # ------------------------------------------------------------------
    # Public API
    def submit(self, order: Order) -> Order:
        """Submit an order to the simulator."""

        if order.order_type not in {OrderType.LIMIT, OrderType.MARKET}:
            raise ValueError(f"Unsupported order type: {order.order_type}")
        if order.order_type is OrderType.LIMIT and (order.price is None or order.price <= 0):
            raise ValueError("Limit orders require a positive price")

        order_id = order.order_id or self._generate_order_id()
        if order.order_id is None:
            order.mark_submitted(order_id)
        self._orders[order_id] = order

        book = self._books.setdefault(order.symbol, _SymbolBook())
        taker_side, maker_side = self._resolve_sides(order.side, book)
        taker_order = _BookOrder(order)
        self._match(taker_order, maker_side)

        if order.remaining_quantity > 1e-12 and order.order_type is OrderType.LIMIT:
            price = float(order.price)
            taker_side.add(price, taker_order)
            self._order_index[order_id] = (order.symbol, order.side, price)
        elif order.remaining_quantity > 1e-12:
            order.cancel()
            self._events.append(
                CancelEvent(order_id=order_id, reason="Unfilled market remainder", timestamp=self._clock())
            )
        return order

    def cancel(self, order_id: str, *, reason: str | None = None) -> bool:
        entry = self._order_index.get(order_id)
        if entry is None:
            return False
        symbol, side, price = entry
        book = self._books[symbol]
        side_book = book.bids if side is OrderSide.BUY else book.asks
        removed = side_book.remove(price, order_id)
        if removed:
            order = self._orders[order_id]
            order.cancel()
            self._order_index.pop(order_id, None)
            self._events.append(
                CancelEvent(order_id=order_id, reason=reason, timestamp=self._clock())
            )
        return removed

    def get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown order_id: {order_id}") from exc

    def open_orders(self, *, symbol: str | None = None) -> List[Order]:
        if symbol is None:
            return [order for order in self._orders.values() if order.is_active]
        book = self._books.get(symbol)
        if book is None:
            return []
        return list(book.bids.iter_orders()) + list(book.asks.iter_orders())

    def drain_events(self) -> List[FillEvent | CancelEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    # ------------------------------------------------------------------
    # Internals
    def _generate_order_id(self) -> str:
        return f"SIM-{next(self._id_counter):010d}"

    def _resolve_sides(
        self, side: OrderSide, book: _SymbolBook
    ) -> tuple[_SideBook, _SideBook]:
        if side is OrderSide.BUY:
            return book.bids, book.asks
        return book.asks, book.bids

    def _match(self, taker: _BookOrder, maker_book: _SideBook) -> None:
        order = taker.order
        while order.remaining_quantity > 1e-12:
            best_level = maker_book.best()
            if best_level is None:
                return
            best = best_level.peek()
            if best is None:
                maker_book.pop_level_if_empty(best_level.price)
                continue
            if not self._prices_cross(order, best_level.price):
                return
            trade_qty = min(order.remaining_quantity, best.order.remaining_quantity)
            trade_price = best_level.price
            timestamp = self._clock()
            best.order.record_fill(trade_qty, trade_price)
            order.record_fill(trade_qty, trade_price)
            best.refresh()
            taker.refresh()
            self._events.append(
                FillEvent(
                    order_id=order.order_id or "",  # pragma: no cover - defensive guard
                    counterparty_id=best.order.order_id or "",
                    quantity=trade_qty,
                    price=trade_price,
                    liquidity="taker",
                    timestamp=timestamp,
                )
            )
            self._events.append(
                FillEvent(
                    order_id=best.order.order_id or "",
                    counterparty_id=order.order_id or "",
                    quantity=trade_qty,
                    price=trade_price,
                    liquidity="maker",
                    timestamp=timestamp,
                )
            )
            if best.order.remaining_quantity <= 1e-12:
                maker_book.pop_level_if_empty(best_level.price)
                self._order_index.pop(best.order.order_id or "", None)

    def _prices_cross(self, order: Order, best_price: float) -> bool:
        if order.order_type is OrderType.MARKET:
            return True
        if order.side is OrderSide.BUY:
            return order.price is not None and order.price + 1e-12 >= best_price
        return order.price is not None and order.price - 1e-12 <= best_price


