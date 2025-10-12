# SPDX-License-Identifier: MIT
"""Level-2 order book simulator tailored for execution what-if analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

from markets.orderbook.src.core.lob import Order as BookOrder
from markets.orderbook.src.core.lob import PriceTimeOrderBook, Side


def _ensure_side(side: Side | str) -> Side:
    if isinstance(side, Side):
        return side
    return Side(str(side).lower())


@dataclass(slots=True)
class BookExecution:
    """Execution report produced by :class:`LevelTwoOrderBookSimulator`."""

    order_id: str
    price: float
    quantity: float
    level_index: int
    queue_position: int
    impacted_price: float
    slippage: float


class LevelTwoOrderBookSimulator:
    """Thin wrapper around :class:`PriceTimeOrderBook` with convenience helpers."""

    def __init__(self) -> None:
        self._book = PriceTimeOrderBook()
        self._order_seq = 0

    # ------------------------------------------------------------------
    # Snapshot management
    def reset(self) -> None:
        """Reset the simulator to an empty state."""

        self._book = PriceTimeOrderBook()
        self._order_seq = 0

    def load_snapshot(
        self,
        bids: Sequence[Tuple[float, float]] | Iterable[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]] | Iterable[Tuple[float, float]],
        *,
        timestamp: int = 0,
    ) -> None:
        """Load an entire snapshot, replacing any existing depth."""

        self.reset()
        for price, qty in bids:
            self.add_limit_order("buy", price, qty, timestamp=timestamp)
        for price, qty in asks:
            self.add_limit_order("sell", price, qty, timestamp=timestamp)

    # ------------------------------------------------------------------
    # Order entry helpers
    def _next_id(self, prefix: str) -> str:
        self._order_seq += 1
        return f"{prefix}-{self._order_seq:08d}"

    def add_limit_order(
        self,
        side: Side | str,
        price: float,
        quantity: float,
        *,
        timestamp: int = 0,
        order_id: str | None = None,
    ) -> str:
        """Insert a new limit order and return its identifier."""

        book_side = _ensure_side(side)
        identifier = order_id or self._next_id("snap")
        self._book.add_limit_order(
            BookOrder(order_id=identifier, side=book_side, price=price, quantity=quantity, timestamp=timestamp)
        )
        return identifier

    def cancel(self, order_id: str) -> bool:
        """Cancel a resting order."""

        return self._book.cancel(order_id)

    # ------------------------------------------------------------------
    # Market order execution
    def execute_market_order(self, side: Side | str, quantity: float) -> List[BookExecution]:
        """Match a market order and return execution reports."""

        book_side = _ensure_side(side)
        executions = self._book.match_market_order(book_side, quantity)
        return [
            BookExecution(
                order_id=fill.order_id,
                price=fill.price,
                quantity=fill.quantity,
                level_index=fill.level_index,
                queue_position=fill.queue_position,
                impacted_price=fill.impacted_price,
                slippage=fill.slippage,
            )
            for fill in executions
        ]

    # ------------------------------------------------------------------
    # Depth utilities
    def best_bid(self) -> float | None:
        return self._book.best_bid()

    def best_ask(self) -> float | None:
        return self._book.best_ask()

    def spread(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return max(0.0, ask - bid)

    def depth(self, side: Side | str | None = None) -> List[Tuple[float, float]]:
        """Return aggregated depth for the selected side or both."""

        if side is None:
            bids = self._book.depth(Side.BUY)
            asks = self._book.depth(Side.SELL)
            # Normalise ordering: descending bids, ascending asks
            bids.sort(reverse=True)
            asks.sort()
            return bids + asks
        book_side = _ensure_side(side)
        entries = self._book.depth(book_side)
        entries.sort(reverse=book_side is Side.BUY)
        return entries

    def iter_depth(self) -> Iterator[Tuple[str, float, float]]:
        """Yield labelled depth entries suitable for reporting."""

        for price, qty in self._book.depth(Side.BUY):
            yield "bid", price, qty
        for price, qty in self._book.depth(Side.SELL):
            yield "ask", price, qty


__all__ = ["BookExecution", "LevelTwoOrderBookSimulator"]
