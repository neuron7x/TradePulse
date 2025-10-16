from __future__ import annotations

import pytest

from interfaces.execution.common import IdempotencyStore


@pytest.mark.unit
def test_idempotency_store_returns_consistent_order_id() -> None:
    store = IdempotencyStore()
    key = "order-123"
    original = {"order_id": "abc-001", "status": "accepted"}
    store.put(key, original)

    def fetcher(record):
        assert record == original
        return {"order_id": "abc-001", "status": "filled"}

    reconciled = store.reconcile(key, fetcher)
    assert reconciled is not None
    assert reconciled["order_id"] == original["order_id"]
    assert store.get(key)["order_id"] == original["order_id"]


@pytest.mark.unit
def test_idempotency_store_detects_remote_divergence() -> None:
    store = IdempotencyStore()
    key = "order-456"
    store.put(key, {"order_id": "legacy", "status": "accepted"})

    def fetcher(record):
        return {"order_id": "replacement", "status": "accepted"}

    reconciled = store.reconcile(key, fetcher)
    assert reconciled is not None
    assert reconciled["order_id"] == "replacement"
    # Local state should be updated to reflect the remote canonical record.
    assert store.get(key)["order_id"] == "replacement"
