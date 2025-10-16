from __future__ import annotations

import sqlite3

import pytest

from execution.risk import KillSwitch, SQLiteKillSwitchStateStore


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
