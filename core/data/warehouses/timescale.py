"""TimescaleDB integration for tick and rollup storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from psycopg import Connection
from psycopg.rows import tuple_row

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
class TimescaleConfig:
    """Configuration parameters for Timescale storage."""

    schema: str = "public"
    raw_table: str = "raw_ticks"
    rollup_table: str = "minute_bars"
    retention_days: int = 30
    rollup_retention_days: int = 180
    chunk_interval_hours: int = 24


class TimescaleWarehouse(TimeSeriesWarehouse):
    """Encapsulates DDL, ingestion and maintenance for TimescaleDB."""

    def __init__(
        self,
        connection: Connection,
        *,
        config: TimescaleConfig | None = None,
        batch_size: int = 5_000,
    ) -> None:
        self._connection = connection
        self._config = config or TimescaleConfig()
        self._batch_size = batch_size

    # -- DDL -----------------------------------------------------------------
    def bootstrap_statements(self) -> Sequence[WarehouseStatement]:
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        raw_table = f"{schema_prefix}{cfg.raw_table}"
        rollup_table = f"{schema_prefix}{cfg.rollup_table}"
        return (
            WarehouseStatement(
                "enable extensions",
                "CREATE EXTENSION IF NOT EXISTS timescaledb;\n"
                "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
            ),
            WarehouseStatement(
                "create raw tick table",
                (
                    f"CREATE TABLE IF NOT EXISTS {raw_table} (\n"
                    "    ts TIMESTAMPTZ NOT NULL,\n"
                    "    symbol TEXT NOT NULL,\n"
                    "    venue TEXT NOT NULL,\n"
                    "    instrument_type TEXT NOT NULL,\n"
                    "    price NUMERIC(18,10) NOT NULL,\n"
                    "    volume NUMERIC(18,10) NOT NULL,\n"
                    "    trade_id TEXT,\n"
                    "    ingest_id UUID NOT NULL DEFAULT gen_random_uuid(),\n"
                    "    ingest_ts TIMESTAMPTZ NOT NULL DEFAULT now()\n"
                    ");"
                ),
            ),
            WarehouseStatement(
                "convert raw table to hypertable",
                (
                    f"SELECT create_hypertable('{raw_table}', 'ts',"
                    " partitioning_column => 'symbol',"
                    f" chunk_time_interval => INTERVAL '{cfg.chunk_interval_hours} hour',"
                    " if_not_exists => TRUE);"
                ),
            ),
            WarehouseStatement(
                "raw table indexes and policies",
                (
                    f"CREATE INDEX IF NOT EXISTS {cfg.raw_table}_symbol_ts_idx\n"
                    f"    ON {raw_table} (symbol, ts DESC);\n"
                    f"SELECT add_retention_policy('{raw_table}', INTERVAL '{cfg.retention_days} days');\n"
                    f"ALTER TABLE {raw_table} SET (timescaledb.compress);\n"
                    f"ALTER TABLE {raw_table} SET (timescaledb.compress_segmentby = 'symbol');\n"
                    f"SELECT add_compression_policy('{raw_table}', INTERVAL '7 days');"
                ),
            ),
            WarehouseStatement(
                "create rollup table",
                (
                    f"CREATE MATERIALIZED VIEW IF NOT EXISTS {rollup_table}\n"
                    "WITH (timescaledb.continuous) AS\n"
                    f"SELECT time_bucket('1 minute', ts) AS window_start,\n"
                    "       symbol,\n"
                    "       venue,\n"
                    "       instrument_type,\n"
                    "       first(price, ts) AS open_price,\n"
                    "       max(price) AS high_price,\n"
                    "       min(price) AS low_price,\n"
                    "       last(price, ts) AS close_price,\n"
                    "       sum(volume) AS volume,\n"
                    "       count(*) AS trade_count\n"
                    f"FROM {raw_table}\n"
                    "GROUP BY window_start, symbol, venue, instrument_type;"
                ),
            ),
            WarehouseStatement(
                "rollup policy",
                (
                    f"SELECT add_continuous_aggregate_policy('{rollup_table}',\n"
                    "    start_offset => INTERVAL '2 days',\n"
                    "    end_offset => INTERVAL '5 minutes',\n"
                    "    schedule_interval => INTERVAL '1 minute');\n"
                    f"SELECT add_retention_policy('{rollup_table}', INTERVAL '{cfg.rollup_retention_days} days');"
                ),
            ),
        )

    def rollup_jobs(self) -> Sequence[RollupJob]:
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        rollup_table = f"{schema_prefix}{cfg.rollup_table}"
        statement = WarehouseStatement(
            "refresh continuous aggregate",
            f"CALL refresh_continuous_aggregate('{rollup_table}', NULL, NULL);",
        )
        return (
            RollupJob(
                name="timescale-minute-bars-refresh",
                statement=statement,
                schedule_hint="*/5 * * * *",
            ),
        )

    def maintenance_tasks(self) -> Sequence[MaintenanceTask]:
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        raw_table = f"{schema_prefix}{cfg.raw_table}"
        return (
            MaintenanceTask(
                name="timescale-reorder-chunks",
                statement=WarehouseStatement(
                    "reorder chunks by symbol",
                    f"CALL reorder_chunks('{raw_table}', 'symbol, ts DESC');",
                ),
                cadence="daily",
            ),
            MaintenanceTask(
                name="timescale-analyze",
                statement=WarehouseStatement(
                    "analyze hypertable statistics",
                    f"ANALYZE {raw_table};",
                ),
                cadence="daily",
            ),
        )

    def sla_queries(self) -> Sequence[SLAQuery]:
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        raw_table = f"{schema_prefix}{cfg.raw_table}"
        rollup_table = f"{schema_prefix}{cfg.rollup_table}"
        return (
            SLAQuery(
                name="tick_ingest_latency",
                sql=(
                    "SELECT symbol, venue,\n"
                    "       max(ts) AS latest_ts,\n"
                    "       now() - max(ts) AS ingest_lag\n"
                    f"FROM {raw_table}\n"
                    "GROUP BY symbol, venue;"
                ),
                description="Latency between live clock and newest tick",
            ),
            SLAQuery(
                name="rollup_freshness",
                sql=(
                    "SELECT symbol, venue,\n"
                    "       max(window_start) AS latest_window,\n"
                    "       now() - max(window_start) AS lag\n"
                    f"FROM {rollup_table}\n"
                    "GROUP BY symbol, venue;"
                ),
                description="Freshness of continuous aggregates",
            ),
            SLAQuery(
                name="ingest_throughput",
                sql=(
                    "SELECT date_trunc('minute', ingest_ts) AS window,\n"
                    "       count(*) / 60.0 AS ticks_per_second\n"
                    f"FROM {raw_table}\n"
                    "WHERE ingest_ts >= now() - INTERVAL '2 hours'\n"
                    "GROUP BY window\n"
                    "ORDER BY window;"
                ),
                description="Ticks per second aggregated per minute",
            ),
        )

    def benchmark_scenarios(self) -> Sequence[BenchmarkScenario]:
        return (
            BenchmarkScenario(
                name="timescale-binary-copy-40k",
                description="COPY ingest sustaining 40k ticks per second",
                target_qps=40_000,
                concurrency=6,
                dataset_hint="synthetic_ticks_40k",
            ),
            BenchmarkScenario(
                name="timescale-dashboard-rollup",
                description="Query 90 days of minute bars under sub-second latency",
                target_qps=750,
                concurrency=4,
                dataset_hint="minute_bars_90d",
            ),
        )

    def backup_plan(self) -> Sequence[BackupStep]:
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        raw_table = f"{schema_prefix}{cfg.raw_table}"
        rollup_table = f"{schema_prefix}{cfg.rollup_table}"
        return (
            BackupStep(
                description="Perform base backup using pg_basebackup",
                command="pg_basebackup -h $PGHOST -D /backups/timescale -U $PGUSER -Fp -Xs -P",
            ),
            BackupStep(
                description="Verify logical checksums for critical tables",
                command=(
                    f"SELECT relname, checksum_failures FROM timescaledb_information.hypertables\n"
                    f"WHERE schema_name = '{cfg.schema}' AND relname IN ('{cfg.raw_table}', '{cfg.rollup_table}');"
                ),
            ),
            BackupStep(
                description="Restore via point-in-time recovery",
                command="pg_ctl restore -D /var/lib/postgresql/data --target='last backup timestamp'",
            ),
        )

    # -- Ingestion ------------------------------------------------------------
    def ingest_ticks(self, ticks: Sequence[PriceTick], *, chunk_size: int | None = None) -> None:
        if not ticks:
            return
        chunk_limit = self._batch_size if chunk_size is None else chunk_size
        if chunk_limit <= 0:
            raise ValueError("chunk_size must be positive")
        cfg = self._config
        schema_prefix = f"{cfg.schema}." if cfg.schema != "public" else ""
        raw_table = f"{schema_prefix}{cfg.raw_table}"
        insert_sql = (
            f"INSERT INTO {raw_table} (ts, symbol, venue, instrument_type, price, volume, trade_id)\n"
            "VALUES (%s, %s, %s, %s, %s, %s, %s);"
        )
        try:
            with self._connection.cursor(row_factory=tuple_row) as cursor:
                for chunk in _chunk_iterable(ticks, chunk_limit):
                    payload = [
                        (
                            tick.timestamp,
                            tick.symbol,
                            tick.venue,
                            tick.instrument_type.value,
                            tick.price,
                            tick.volume,
                            tick.trade_id,
                        )
                        for tick in chunk
                    ]
                    cursor.executemany(insert_sql, payload)
                    _LOGGER.debug(
                        "timescale_ingest_batch",
                        rows=len(payload),
                        table=raw_table,
                    )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise


def _chunk_iterable(items: Iterable[PriceTick], size: int) -> Iterable[Sequence[PriceTick]]:
    batch: list[PriceTick] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


__all__ = ["TimescaleWarehouse", "TimescaleConfig"]
