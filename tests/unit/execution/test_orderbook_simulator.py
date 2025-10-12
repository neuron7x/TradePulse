# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from execution.orderbook import LevelTwoOrderBookSimulator


def test_order_book_snapshot_and_depth() -> None:
    simulator = LevelTwoOrderBookSimulator()
    simulator.load_snapshot(
        bids=[(100.0, 1.0), (99.5, 2.0)],
        asks=[(100.5, 1.5), (101.0, 3.0)],
        timestamp=1,
    )

    assert simulator.best_bid() == pytest.approx(100.0)
    assert simulator.best_ask() == pytest.approx(100.5)
    assert simulator.spread() == pytest.approx(0.5)

    bids = simulator.depth("buy")
    asks = simulator.depth("sell")
    assert bids[0][0] == pytest.approx(100.0)
    assert bids[0][1] == pytest.approx(1.0)
    assert asks[0][0] == pytest.approx(100.5)
    assert asks[0][1] == pytest.approx(1.5)

    inserted_id = simulator.add_limit_order("sell", price=100.4, quantity=0.5, timestamp=2)
    assert simulator.best_ask() == pytest.approx(100.4)
    simulator.cancel(inserted_id)
    assert simulator.best_ask() == pytest.approx(100.5)


def test_order_book_market_execution_consumes_depth() -> None:
    simulator = LevelTwoOrderBookSimulator()
    simulator.load_snapshot(bids=[(99.0, 1.0)], asks=[(100.0, 0.5), (101.0, 1.0)])

    fills = simulator.execute_market_order("buy", 1.0)
    assert len(fills) == 2
    assert fills[0].price == pytest.approx(100.0)
    assert fills[0].quantity == pytest.approx(0.5)
    assert fills[1].price == pytest.approx(101.0)
    assert fills[1].quantity == pytest.approx(0.5)

    remaining = simulator.depth("sell")
    assert remaining[0][0] == pytest.approx(101.0)
    assert remaining[0][1] == pytest.approx(0.5)
    assert simulator.best_bid() == pytest.approx(99.0)
