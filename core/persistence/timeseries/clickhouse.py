"""ClickHouse adaptor for TradePulse time-series data."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from typing import Iterable, Sequence

from .base import TimeSeriesAdapter, TimeSeriesPoint


class ClickHouseTimeSeriesAdapter(TimeSeriesAdapter):
    """Implementation backed by :mod:`clickhouse_connect`."""

    def __init__(
        self,
        host: str,
        *,
        database: str = "default",
        username: str | None = None,
        password: str | None = None,
        secure: bool = True,
        port: int | None = None,
    ) -> None:
        self._client = self._create_client(host, database, username, password, secure, port)
        self._database = database

    def _create_client(self, host: str, database: str, username: str | None, password: str | None, secure: bool, port: int | None):
        spec = importlib.util.find_spec("clickhouse_connect")
        if spec is None:
            msg = "clickhouse-connect package is required for ClickHouseTimeSeriesAdapter"
            raise RuntimeError(msg)
        module = importlib.import_module("clickhouse_connect")
        client = module.get_client(
            host=host,
            database=database,
            username=username,
            password=password,
            secure=secure,
            port=port,
        )
        return client

    def _ensure_table(self, table: str) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            timestamp DateTime64(6) CODEC(DoubleDelta, ZSTD(1)),
            tags Map(String, String),
            values Map(String, Float64)
        )
        ENGINE = MergeTree()
        ORDER BY (timestamp)
        PARTITION BY toDate(timestamp)
        SETTINGS index_granularity = 8192
        """
        self._client.command(ddl)

    def write_points(self, table: str, points: Sequence[TimeSeriesPoint]) -> int:
        if not points:
            return 0
        self._ensure_table(table)
        rows = [
            {
                "timestamp": point.timestamp,
                "tags": point.tags or {},
                "values": {k: float(v) for k, v in point.values.items()},
            }
            for point in points
        ]
        self._client.insert(table, rows)
        return len(points)

    def read_points(
        self,
        table: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterable[TimeSeriesPoint]:
        conditions: list[str] = []
        params: dict[str, datetime] = {}
        if start is not None:
            conditions.append("timestamp >= %(start)s")
            params["start"] = start
        if end is not None:
            conditions.append("timestamp <= %(end)s")
            params["end"] = end
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        query = f"SELECT timestamp, tags, values FROM {table} {where} ORDER BY timestamp ASC {limit_clause}"
        result = self._client.query(query, params=params or None)
        for row in result.result_rows:
            timestamp, tags, values = row
            yield TimeSeriesPoint(timestamp=timestamp, tags=tags, values=values)

    def close(self) -> None:
        self._client.close()
