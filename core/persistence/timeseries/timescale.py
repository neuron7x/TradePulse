"""TimescaleDB implementation of the :class:`TimeSeriesAdapter` protocol."""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator, Sequence

from .base import TimeSeriesAdapter, TimeSeriesPoint


class TimescaleTimeSeriesAdapter(TimeSeriesAdapter):
    """Persist time-series data into TimescaleDB using psycopg."""

    def __init__(
        self,
        dsn: str,
        *,
        hypertable: bool = True,
        chunk_interval: str = "1 day",
        time_column: str = "timestamp",
    ) -> None:
        self._dsn = dsn
        self._hypertable = hypertable
        self._chunk_interval = chunk_interval
        self._time_column = time_column
        self._psycopg = self._load_driver()
        self._connection = None

    def _load_driver(self):
        spec = importlib.util.find_spec("psycopg")
        if spec is None:
            msg = "psycopg package is required to use TimescaleTimeSeriesAdapter"
            raise RuntimeError(msg)
        module = importlib.import_module("psycopg")
        return module

    def connect(self) -> None:
        if self._connection is None:
            self._connection = self._psycopg.connect(self._dsn, autocommit=True)

    @contextmanager
    def _cursor(self) -> Iterator:
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        with self._connection.cursor() as cursor:
            yield cursor

    def _ensure_table(self, table: str, sample: TimeSeriesPoint) -> None:
        create_stmt = (
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f"{self._time_column} TIMESTAMPTZ NOT NULL,"
            " tags JSONB DEFAULT '{}'::jsonb,"
            " values JSONB NOT NULL,"
            f" PRIMARY KEY ({self._time_column}, tags))"
        )
        with self._cursor() as cur:
            cur.execute(create_stmt)
            if self._hypertable:
                cur.execute(
                    "SELECT create_hypertable(%s, %s, if_not_exists => true, chunk_time_interval => %s)",
                    (table, self._time_column, self._chunk_interval),
                )

    def write_points(self, table: str, points: Sequence[TimeSeriesPoint]) -> int:
        if not points:
            return 0
        self._ensure_table(table, points[0])
        insert_stmt = (
            f"INSERT INTO {table} ({self._time_column}, tags, values) "
            "VALUES (%s, %s::jsonb, %s::jsonb) "
            "ON CONFLICT ({time_column}, tags) DO UPDATE SET values = excluded.values"
        ).format(time_column=self._time_column)
        payload = [
            (
                point.timestamp,
                (point.tags or {}),
                point.values,
            )
            for point in points
        ]
        with self._cursor() as cur:
            cur.executemany(insert_stmt, payload)
        return len(points)

    def read_points(
        self,
        table: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterable[TimeSeriesPoint]:
        clauses = []
        params: list = []
        if start is not None:
            clauses.append(f"{self._time_column} >= %s")
            params.append(start)
        if end is not None:
            clauses.append(f"{self._time_column} <= %s")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
        query = (
            f"SELECT {self._time_column}, tags, values FROM {table} {where} "
            f"ORDER BY {self._time_column} ASC {limit_clause}"
        )
        with self._cursor() as cur:
            cur.execute(query, params or None)
            rows = cur.fetchall()
        for timestamp, tags, values in rows:
            yield TimeSeriesPoint(timestamp=timestamp, tags=tags, values=values)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
