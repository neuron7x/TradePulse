from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from execution.risk import (
    DataQualityError,
    KillSwitch,
    PostgresKillSwitchStateStore,
    SQLiteKillSwitchStateStore,
)


class _InMemoryKillSwitchDB:
    """In-memory simulation of the PostgreSQL kill-switch table."""

    def __init__(self) -> None:
        self.state: tuple[bool, str, datetime] | None = None
        self.failures: deque[Exception] = deque()
        self.lock = threading.Lock()


class _FakeCursor:
    def __init__(self, database: _InMemoryKillSwitchDB) -> None:
        self._database = database
        self._result: tuple[bool, str, datetime] | None = None
        self.rowcount = -1

    def execute(self, query: str, params: object | None = None) -> None:
        if self._database.failures:
            failure = self._database.failures.popleft()
            raise failure

        normalised = " ".join(query.split()).lower()
        with self._database.lock:
            if normalised.startswith("create table"):
                self.rowcount = 0
                return
            if normalised.startswith("select"):
                self._result = self._database.state
                self.rowcount = 0 if self._result is None else 1
                return
            if normalised.startswith("insert into kill_switch_state"):
                if isinstance(params, dict):
                    engaged = bool(params.get("engaged", False))
                    reason = params.get("reason", "") or ""
                else:  # pragma: no cover - defensive guard
                    engaged = bool(params[1])
                    reason = params[2]
                timestamp = datetime.now(timezone.utc)
                self._database.state = (engaged, reason, timestamp)
                self.rowcount = 1
                return
        raise AssertionError(f"Unsupported query in fake cursor: {query}")

    def fetchone(self) -> tuple[bool, str, datetime] | None:
        return self._result

    def fetchall(self) -> list[tuple[bool, str, datetime]]:
        return [] if self._result is None else [self._result]

    def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(self, database: _InMemoryKillSwitchDB) -> None:
        self._database = database
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._database)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def postgres_store() -> tuple[PostgresKillSwitchStateStore, _InMemoryKillSwitchDB]:
    database = _InMemoryKillSwitchDB()

    def connection_factory() -> _FakeConnection:
        return _FakeConnection(database)

    store = PostgresKillSwitchStateStore(
        "postgresql://test",  # DSN is not used by the fake connection factory.
        tls=None,
        pool_min_size=0,
        pool_max_size=4,
        acquire_timeout=0.01,
        max_retries=2,
        retry_interval=0.0,
        backoff_multiplier=1.0,
        connection_factory=connection_factory,
    )

    try:
        yield store, database
    finally:
        store.close()


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


def test_sqlite_store_retries_when_locked(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_sqlite_store_raises_when_lock_persists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    stale_timestamp = (current_time - timedelta(seconds=5)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
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


def test_postgres_store_persists_state(postgres_store) -> None:
    store, _ = postgres_store
    kill_switch = KillSwitch(store=store)
    assert kill_switch.is_triggered() is False

    kill_switch.trigger("postgres engaged")
    assert store.load() == (True, "postgres engaged")

    restored = KillSwitch(store=store)
    assert restored.is_triggered() is True
    assert restored.reason == "postgres engaged"


def test_postgres_store_enforces_reason_contracts(postgres_store) -> None:
    store, _ = postgres_store

    with pytest.raises(DataQualityError):
        store.save(True, "")

    with pytest.raises(DataQualityError):
        store.save(True, "\x00invalid")

    store.save(True, "ha event")
    assert store.load() == (True, "ha event")


def test_postgres_store_detects_stale_state(postgres_store) -> None:
    store, database = postgres_store
    store.save(False, "")

    with database.lock:
        database.state = (False, "", datetime.now(timezone.utc) - timedelta(hours=2))

    with pytest.raises(DataQualityError):
        store.load()


def test_postgres_store_retries_on_transient_errors(postgres_store) -> None:
    store, database = postgres_store
    database.failures.append(ConnectionError("primary down"))

    store.save(True, "ha failover")
    assert store.load() == (True, "ha failover")


@pytest.mark.anyio
async def test_postgres_store_handles_concurrent_clients(postgres_store) -> None:
    store, _ = postgres_store

    async def engage(idx: int) -> tuple[bool, str] | None:
        reason = f"maintenance-{idx}"
        await asyncio.to_thread(store.save, True, reason)
        return await asyncio.to_thread(store.load)

    results = await asyncio.gather(*(engage(i) for i in range(3)))

    assert any(
        result == (True, f"maintenance-{idx}") for idx, result in enumerate(results)
    )
    assert store.load()[0] is True
