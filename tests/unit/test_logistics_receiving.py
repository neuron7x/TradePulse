from datetime import date

import pytest

from core.logistics import (
    QualityStatus,
    ReceivedLine,
    ReceivingSession,
    WarehouseInventory,
)


def test_receiving_session_tracks_shortages_and_quality_breakdown():
    session = ReceivingSession({"A": 10, "B": 5})
    session.accept("A", 6, lot="LOT-A", expiration=date(2024, 12, 31))
    session.accept("B", 4)
    session.reject("B", 1, reason=QualityStatus.DAMAGED)

    summary = session.summary()

    assert summary["accepted"] == {"A": 6, "B": 4}
    assert summary["rejected"] == {"B": {QualityStatus.DAMAGED: 1}}
    assert summary["shortages"] == {"A": 4, "B": 1}
    assert summary["overages"] == {}
    assert summary["unplanned"] == {}
    assert summary["complete"] is False


def test_receiving_session_detects_overages_and_unplanned():
    session = ReceivingSession({"A": 5})
    session.accept("A", 7)
    session.accept("C", 3)

    summary = session.summary()

    assert summary["accepted"] == {"A": 7, "C": 3}
    assert summary["overages"] == {"A": 2}
    assert summary["unplanned"] == {"C": 3}
    assert summary["shortages"] == {}
    assert summary["complete"] is True


def test_receiving_session_reject_helper_and_validation():
    session = ReceivingSession({"A": 5})
    session.reject("A", 2, reason=QualityStatus.DAMAGED, notes="broken packaging")

    rejected = session.rejected_items()
    assert rejected == {"A": {QualityStatus.DAMAGED: 2}}
    assert session.accepted_items() == {}

    with pytest.raises(ValueError):
        session.reject("A", 1, reason=QualityStatus.ACCEPTED)

    with pytest.raises(ValueError):
        session.accept("A", 0)

    with pytest.raises(ValueError):
        ReceivedLine("", 1)


def test_inventory_updates_only_with_accepted_items():
    inventory = WarehouseInventory({"A": 5})
    session = ReceivingSession({"A": 2, "B": 3})
    session.accept("A", 2)
    session.accept("B", 3)
    session.reject("B", 1, reason=QualityStatus.EXPIRED)

    inventory.apply_receiving(session)

    assert inventory.quantity("A") == 7
    assert inventory.quantity("B") == 3
    assert inventory.quantity("C") == 0
    assert inventory.snapshot() == {"A": 7, "B": 3}


def test_iter_all_lines_preserves_insert_order():
    session = ReceivingSession({"A": 1})
    session.accept("A", 1, notes="first")
    session.reject("A", 2, reason=QualityStatus.DAMAGED, notes="second")

    lines = list(session.iter_all_lines())
    assert [line.notes for line in lines] == ["first", "second"]
