from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from core.data.models import InstrumentType, MarketMetadata, PriceTick
from core.data.warehouses import ClickHouseConfig, ClickHouseWarehouse, TimescaleWarehouse


class RecordingResponse:
    def __init__(self, status_code: int = 200, text: str = "OK") -> None:
        self.status_code = status_code
        self.text = text


class RecordingClient:
    def __init__(self, *, status_code: int = 200, text: str = "OK") -> None:
        self.calls: list[dict] = []
        self._status_code = status_code
        self._text = text

    def post(self, url: str, *, params: dict | None = None, content: bytes | None = None, headers=None, timeout=None):
        record = {
            "url": url,
            "params": params or {},
            "content": content or b"",
            "headers": headers or {},
            "timeout": timeout,
        }
        self.calls.append(record)
        return RecordingResponse(self._status_code, self._text)


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[tuple]] | tuple[str, tuple]] = []
        self.closed = False

    def executemany(self, sql: str, params):
        self.executed.append((sql, list(params)))

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self) -> None:
        self.cursors: list[FakeCursor] = []
        self.committed = False
        self.rolled_back = False

    def cursor(self, row_factory=None) -> FakeCursor:
        cursor = FakeCursor()
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


@pytest.fixture
def sample_ticks() -> list[PriceTick]:
    metadata = MarketMetadata(
        symbol="BTCUSDT",
        venue="binance",
        instrument_type=InstrumentType.SPOT,
    )
    return [
        PriceTick(
            metadata=metadata,
            timestamp=datetime(2024, 1, 1, 0, 0, i, tzinfo=timezone.utc),
            price=Decimal("30000.0") + Decimal(i),
            volume=Decimal("0.1"),
            trade_id=f"trade-{i}",
        )
        for i in range(3)
    ]


def test_clickhouse_bootstrap_contains_ttl() -> None:
    client = RecordingClient()
    warehouse = ClickHouseWarehouse(client, config=ClickHouseConfig(retention_days=45))

    ddl = warehouse.bootstrap_statements()
    raw_statement = next(stmt for stmt in ddl if "raw tick" in stmt.description)

    assert "TTL ts + INTERVAL 45 DAY" in raw_statement.sql
    mv_statement = next(stmt for stmt in ddl if "materialized view" in stmt.description)
    assert "toStartOfInterval" in mv_statement.sql
    assert "argMax(price" in mv_statement.sql


def test_clickhouse_ingest_batches(sample_ticks: list[PriceTick]) -> None:
    client = RecordingClient()
    warehouse = ClickHouseWarehouse(client)

    warehouse.ingest_ticks(sample_ticks, chunk_size=2)

    assert len(client.calls) == 2
    first_payload = client.calls[0]["content"].decode("utf-8")
    assert "BTC/USDT" in first_payload
    assert "\n" in first_payload  # multi-row payload should be newline delimited


def test_clickhouse_ingest_failure_raises(sample_ticks: list[PriceTick]) -> None:
    client = RecordingClient(status_code=500, text="boom")
    warehouse = ClickHouseWarehouse(client)

    with pytest.raises(RuntimeError):
        warehouse.ingest_ticks(sample_ticks[:1])


def test_clickhouse_sla_queries() -> None:
    client = RecordingClient()
    warehouse = ClickHouseWarehouse(client)

    queries = {query.name: query.sql for query in warehouse.sla_queries()}
    assert "tick_ingest_latency" in queries
    assert "ticks_per_second" in queries["ingest_throughput"]


def test_timescale_bootstrap_contains_policies() -> None:
    connection = FakeConnection()
    warehouse = TimescaleWarehouse(connection)

    ddl = warehouse.bootstrap_statements()
    policy_stmt = next(stmt for stmt in ddl if "indexes and policies" in stmt.description)
    assert "add_retention_policy" in policy_stmt.sql
    assert "add_compression_policy" in policy_stmt.sql
    rollup_stmt = next(stmt for stmt in ddl if "rollup policy" in stmt.description)
    assert "continuous_aggregate_policy" in rollup_stmt.sql


def test_timescale_ingest_chunks_and_commit(sample_ticks: list[PriceTick]) -> None:
    connection = FakeConnection()
    warehouse = TimescaleWarehouse(connection)

    warehouse.ingest_ticks(sample_ticks, chunk_size=2)

    assert connection.committed is True
    cursor = connection.cursors[0]
    assert len(cursor.executed) == 2
    insert_sql, first_batch = cursor.executed[0]
    assert "INSERT INTO" in insert_sql
    assert first_batch[0][1] == "BTC/USDT"


def test_timescale_ingest_invalid_chunk_size(sample_ticks: list[PriceTick]) -> None:
    connection = FakeConnection()
    warehouse = TimescaleWarehouse(connection)

    with pytest.raises(ValueError):
        warehouse.ingest_ticks(sample_ticks, chunk_size=0)


def test_timescale_ingest_rolls_back_on_failure(sample_ticks: list[PriceTick]) -> None:
    class ErrorCursor(FakeCursor):
        def executemany(self, sql: str, params):  # type: ignore[override]
            raise RuntimeError("boom")

    class ErrorConnection(FakeConnection):
        def cursor(self, row_factory=None):  # type: ignore[override]
            cursor = ErrorCursor()
            self.cursors.append(cursor)
            return cursor

    connection = ErrorConnection()
    warehouse = TimescaleWarehouse(connection)

    with pytest.raises(RuntimeError):
        warehouse.ingest_ticks(sample_ticks[:1])
    assert connection.rolled_back is True


def test_backup_plans_cover_both_systems() -> None:
    client = RecordingClient()
    clickhouse = ClickHouseWarehouse(client)
    timescale = TimescaleWarehouse(FakeConnection())

    ch_backup = " ".join(step.command for step in clickhouse.backup_plan())
    ts_backup = " ".join(step.command for step in timescale.backup_plan())

    assert "BACKUP TABLE" in ch_backup
    assert "pg_basebackup" in ts_backup
    assert "restore" in ts_backup.lower()
