from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from execution.risk import DataQualityError, KillSwitch, SQLiteKillSwitchStateStore


def test_kill_switch_persists_state_across_instances(tmp_path) -> None:
    store_path = tmp_path / "state" / "kill_switch.sqlite"
    store = SQLiteKillSwitchStateStore(store_path)

    first = KillSwitch(store=store)
    assert first.is_triggered() is False

    first.trigger("incident response")
    assert store.load() == (True, "incident response")

    restored = KillSwitch(store=store)
    assert restored.is_triggered() is True
    assert restored.reason == "incident response"


def test_kill_switch_reset_persists_clear_state(tmp_path) -> None:
    store = SQLiteKillSwitchStateStore(tmp_path / "kill_switch.sqlite")
    kill_switch = KillSwitch(store=store)
    kill_switch.trigger("maintenance")
    kill_switch.reset()

    persisted = store.load()
    assert persisted == (False, "")

    reloaded = KillSwitch(store=store)
    assert reloaded.is_triggered() is False
    assert reloaded.reason == ""


def test_kill_switch_refreshes_from_store_between_instances(tmp_path) -> None:
    store = SQLiteKillSwitchStateStore(tmp_path / "shared.sqlite")
    primary = KillSwitch(store=store)
    secondary = KillSwitch(store=store)

    assert primary.is_triggered() is False
    assert secondary.is_triggered() is False

    secondary.trigger("other worker engaged")

    assert primary.is_triggered() is True
    assert primary.reason == "other worker engaged"

    secondary.reset()

    assert primary.is_triggered() is False
    assert primary.reason == ""


def test_sqlite_store_retries_when_locked(tmp_path, monkeypatch) -> None:
    store = SQLiteKillSwitchStateStore(
        tmp_path / "retryable.sqlite",
        max_retries=3,
        retry_interval=0.001,
    )

    original_connect = sqlite3.connect
    call_state = {"count": 0}

    def flaky_connect(*args, **kwargs):
        if call_state["count"] < 2:
            call_state["count"] += 1
            raise sqlite3.OperationalError("database is locked")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", flaky_connect)

    store.save(True, "lock-step")

    assert call_state["count"] == 2
    assert store.load() == (True, "lock-step")


def test_sqlite_store_raises_when_lock_persists(tmp_path, monkeypatch) -> None:
    store = SQLiteKillSwitchStateStore(
        tmp_path / "permanent_lock.sqlite",
        max_retries=1,
        retry_interval=0.001,
    )

    def always_locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", always_locked)

    with pytest.raises(sqlite3.OperationalError):
        store.save(True, "doomed")


def test_sqlite_store_enforces_staleness_contract(tmp_path) -> None:
    current_time = datetime.now(timezone.utc)

    def clock() -> datetime:
        return current_time

    store_path = tmp_path / "staleness.sqlite"
    store = SQLiteKillSwitchStateStore(
        store_path,
        max_staleness=1.0,
        max_future_drift=3600.0,
        clock=clock,
    )

    store.save(True, "initial trip")

    stale_timestamp = (current_time - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE kill_switch_state SET updated_at = ? WHERE id = 1",
            (stale_timestamp,),
        )

    with pytest.raises(DataQualityError) as excinfo:
        store.load()

    assert "stale" in str(excinfo.value).lower()
    assert store.is_quarantined() is True
    assert store.quarantine_reason() is not None

    with pytest.raises(DataQualityError):
        store.save(False, "")

    store.clear_quarantine()
    current_time = datetime.now(timezone.utc)
    store.save(False, "")
    current_time = datetime.now(timezone.utc) - timedelta(milliseconds=100)

    assert store.load() == (False, "")


def test_sqlite_store_enforces_reason_quality(tmp_path) -> None:
    store = SQLiteKillSwitchStateStore(tmp_path / "reason.sqlite")

    with pytest.raises(DataQualityError):
        store.save(True, "")

    with pytest.raises(DataQualityError):
        store.save(True, "\x00invalid")

    acceptable = "ok"
    store.save(True, acceptable)
    assert store.load() == (True, acceptable)


def test_sqlite_store_detects_reason_length_anomaly(tmp_path) -> None:
    store_path = tmp_path / "length.sqlite"
    store = SQLiteKillSwitchStateStore(store_path)
    store.save(False, "")

    anomalous_reason = "x" * 600

    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE kill_switch_state SET reason = ? WHERE id = 1",
            (anomalous_reason,),
        )

    with pytest.raises(DataQualityError):
        store.load()

    assert store.is_quarantined() is True
