"""ClickHouse time-series warehouse integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence
from uuid import uuid4

import httpx

from core.data.models import PriceTick
from core.utils.logging import get_logger

from .base import (
    BackupStep,
    BenchmarkScenario,
    MaintenanceTask,
    RollupJob,
    SLAQuery,
    TimeSeriesWarehouse,
    WarehouseStatement,
)

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ClickHouseConfig:
    """Runtime configuration for the ClickHouse integration."""

    database: str = "tradepulse"
    raw_table: str = "raw_ticks"
    rollup_table: str = "minute_bars"
    retention_days: int = 30
    rollup_retention_days: int = 180
    timezone_name: str = "UTC"
    write_path: str = "/"


class ClickHouseWarehouse(TimeSeriesWarehouse):
    """Manage ClickHouse schemas, ingestion and operational tasks."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        config: ClickHouseConfig | None = None,
        insert_timeout: float = 10.0,
    ) -> None:
        self._client = client
        self._config = config or ClickHouseConfig()
        self._insert_timeout = insert_timeout

    # -- DDL -----------------------------------------------------------------
    def bootstrap_statements(self) -> Sequence[WarehouseStatement]:
        cfg = self._config
        return (
            WarehouseStatement(
                "create database",
                f"CREATE DATABASE IF NOT EXISTS {cfg.database}",
            ),
            WarehouseStatement(
                "create raw tick table",
                (
                    f"CREATE TABLE IF NOT EXISTS {cfg.database}.{cfg.raw_table} (\n"
                    "    ts DateTime64(6, '{cfg.timezone_name}') CODEC(Delta, ZSTD),\n"
                    "    symbol LowCardinality(String),\n"
                    "    venue LowCardinality(String),\n"
                    "    instrument_type LowCardinality(String),\n"
                    "    price Decimal64(10),\n"
                    "    volume Decimal64(10),\n"
                    "    trade_id String DEFAULT '',\n"
                    "    ingest_id UUID DEFAULT generateUUIDv4(),\n"
                    "    ingest_ts DateTime64(6, '{cfg.timezone_name}') DEFAULT now('{cfg.timezone_name}')\n"
                    "    , INDEX idx_symbol symbol TYPE set(0) GRANULARITY 1\n"
                    ")\n"
                    "ENGINE = MergeTree\n"
                    "PARTITION BY toDate(ts)\n"
                    "ORDER BY (symbol, ts)\n"
                    f"TTL ts + INTERVAL {cfg.retention_days} DAY DELETE\n"
                    "SETTINGS index_granularity = 8192, allow_nullable_key = 0\n"
                    "COMMENT 'Tick level market data'"
                ),
            ),
            WarehouseStatement(
                "create minute bar table",
                (
                    f"CREATE TABLE IF NOT EXISTS {cfg.database}.{cfg.rollup_table} (\n"
                    "    window_start DateTime64(6, '{cfg.timezone_name}'),\n"
                    "    symbol LowCardinality(String),\n"
                    "    venue LowCardinality(String),\n"
                    "    instrument_type LowCardinality(String),\n"
                    "    open_price Decimal64(10),\n"
                    "    high_price Decimal64(10),\n"
                    "    low_price Decimal64(10),\n"
                    "    close_price Decimal64(10),\n"
                    "    volume Decimal64(12),\n"
                    "    trade_count UInt64,\n"
                    "    ingest_ts DateTime64(6, '{cfg.timezone_name}') DEFAULT now('{cfg.timezone_name}')\n"
                    "    , INDEX idx_rollup_symbol symbol TYPE set(0) GRANULARITY 1\n"
                    ")\n"
                    "ENGINE = MergeTree\n"
                    "PARTITION BY toYYYYMM(window_start)\n"
                    "ORDER BY (symbol, window_start)\n"
                    f"TTL window_start + INTERVAL {cfg.rollup_retention_days} DAY DELETE\n"
                    "SETTINGS index_granularity = 2048\n"
                    "COMMENT 'One minute rollups from ticks'"
                ),
            ),
            WarehouseStatement(
                "create minute bar materialized view",
                (
                    f"CREATE MATERIALIZED VIEW IF NOT EXISTS {cfg.database}.mv_{cfg.rollup_table} \n"
                    f"TO {cfg.database}.{cfg.rollup_table} AS\n"
                    "SELECT\n"
                    "    toStartOfInterval(ts, INTERVAL 1 MINUTE, '{cfg.timezone_name}') AS window_start,\n"
                    "    symbol,\n"
                    "    venue,\n"
                    "    instrument_type,\n"
                    "    argMin(price, ts) AS open_price,\n"
                    "    argMax(price, ts) AS close_price,\n"
                    "    max(price) AS high_price,\n"
                    "    min(price) AS low_price,\n"
                    "    sum(volume) AS volume,\n"
                    "    count() AS trade_count\n"
                    f"FROM {cfg.database}.{cfg.raw_table}\n"
                    "GROUP BY\n"
                    "    window_start, symbol, venue, instrument_type"
                ),
            ),
        )

    def rollup_jobs(self) -> Sequence[RollupJob]:
        cfg = self._config
        statement = WarehouseStatement(
            "force refresh materialized view",
            f"OPTIMIZE TABLE {cfg.database}.mv_{cfg.rollup_table} FINAL",
        )
        return (
            RollupJob(
                name="clickhouse-minute-bars-refresh",
                statement=statement,
                schedule_hint="*/5 * * * *",
            ),
        )

    def maintenance_tasks(self) -> Sequence[MaintenanceTask]:
        cfg = self._config
        return (
            MaintenanceTask(
                name="clickhouse-raw-optimize",
                statement=WarehouseStatement(
                    "compact raw tick partitions",
                    f"OPTIMIZE TABLE {cfg.database}.{cfg.raw_table} FINAL DEDUPLICATE",
                ),
                cadence="hourly",
            ),
            MaintenanceTask(
                name="clickhouse-rollup-optimize",
                statement=WarehouseStatement(
                    "compact rollup partitions",
                    f"OPTIMIZE TABLE {cfg.database}.{cfg.rollup_table} FINAL",
                ),
                cadence="daily",
            ),
        )

    def sla_queries(self) -> Sequence[SLAQuery]:
        cfg = self._config
        return (
            SLAQuery(
                name="tick_ingest_latency",
                sql=(
                    "SELECT symbol, venue, \n"
                    "       max(ts) AS latest_ts, \n"
                    "       now() - max(ts) AS ingest_lag\n"
                    f"FROM {cfg.database}.{cfg.raw_table}\n"
                    "GROUP BY symbol, venue"
                ),
                description="Latency between newest tick and current time",
            ),
            SLAQuery(
                name="minute_bar_freshness",
                sql=(
                    "SELECT symbol, venue,\n"
                    "       max(window_start) AS latest_window,\n"
                    "       now() - max(window_start) AS lag\n"
                    f"FROM {cfg.database}.{cfg.rollup_table}\n"
                    "GROUP BY symbol, venue"
                ),
                description="Freshness of minute rollups",
            ),
            SLAQuery(
                name="ingest_throughput",
                sql=(
                    "SELECT toStartOfInterval(ingest_ts, INTERVAL 5 MINUTE) AS window,\n"
                    "       count() / 300 AS ticks_per_second\n"
                    f"FROM {cfg.database}.{cfg.raw_table}\n"
                    "WHERE ingest_ts >= now() - INTERVAL 2 HOUR\n"
                    "GROUP BY window\n"
                    "ORDER BY window"
                ),
                description="Ingestion throughput over the last two hours",
            ),
        )

    def benchmark_scenarios(self) -> Sequence[BenchmarkScenario]:
        return (
            BenchmarkScenario(
                name="clickhouse-tick-ingest-50k",
                description="Sustain 50k ticks/s via JSONEachRow inserts",
                target_qps=50_000,
                concurrency=8,
                dataset_hint="synthetic_ticks_50k",
            ),
            BenchmarkScenario(
                name="clickhouse-rollup-scan",
                description="Scan 180 days of rollups for dashboard workloads",
                target_qps=1_000,
                concurrency=4,
                dataset_hint="minute_bars_180d",
            ),
        )

    def backup_plan(self) -> Sequence[BackupStep]:
        cfg = self._config
        return (
            BackupStep(
                description="Snapshot raw and rollup tables to S3",
                command=(
                    f"BACKUP TABLE {cfg.database}.{cfg.raw_table}, {cfg.database}.{cfg.rollup_table} "
                    "TO 's3://tradepulse-clickhouse-backups/{date}/' SETTINGS compression='zstd'"
                ),
            ),
            BackupStep(
                description="Validate backup metadata",
                command="SYSTEM RESTORE FROM 's3://tradepulse-clickhouse-backups/{date}/' DRY RUN",
            ),
        )

    # -- Ingestion ------------------------------------------------------------
    def ingest_ticks(self, ticks: Sequence[PriceTick], *, chunk_size: int = 10_000) -> None:
        if not ticks:
            return
        cfg = self._config
        table = f"{cfg.database}.{cfg.raw_table}"
        insert_query = f"INSERT INTO {table} FORMAT JSONEachRow"
        for chunk in _chunk_iterable(ticks, chunk_size):
            payload = "\n".join(self._serialise_tick(tick) for tick in chunk)
            response = self._client.post(
                cfg.write_path,
                params={"query": insert_query},
                content=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=self._insert_timeout,
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"ClickHouse ingest failed with status {response.status_code}: {response.text}"
                )
            _LOGGER.debug(
                "clickhouse_ingest_batch", rows=len(chunk), table=table, status=response.status_code
            )

    def ingest_bars(
        self, bars: Iterable[dict], *, chunk_size: int = 2_000
    ) -> None:  # pragma: no cover - exercised in integration
        rows = list(bars)
        if not rows:
            return
        cfg = self._config
        table = f"{cfg.database}.{cfg.rollup_table}"
        insert_query = f"INSERT INTO {table} FORMAT JSONEachRow"
        for chunk in _chunk_iterable(rows, chunk_size):
            payload = "\n".join(json.dumps(row, separators=(",", ":")) for row in chunk)
            response = self._client.post(
                cfg.write_path,
                params={"query": insert_query},
                content=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=self._insert_timeout,
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"ClickHouse bar ingest failed with status {response.status_code}: {response.text}"
                )
            _LOGGER.debug(
                "clickhouse_bar_ingest_batch",
                rows=len(chunk),
                table=table,
                status=response.status_code,
            )

    def _serialise_tick(self, tick: PriceTick) -> str:
        payload = {
            "ts": tick.timestamp.astimezone(timezone.utc).isoformat(),
            "symbol": tick.symbol,
            "venue": tick.venue,
            "instrument_type": tick.instrument_type.value,
            "price": str(tick.price),
            "volume": str(tick.volume),
            "trade_id": tick.trade_id or "",
            "ingest_id": str(uuid4()),
            "ingest_ts": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(payload, separators=(",", ":"))


def _chunk_iterable(items: Iterable, size: int) -> Iterable[Sequence]:
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


__all__ = ["ClickHouseWarehouse", "ClickHouseConfig"]
