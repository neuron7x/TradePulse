from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pytest

from core.persistence.timeseries import TimeSeriesPoint
from core.persistence.timeseries import clickhouse as clickhouse_module
from core.persistence.timeseries import parquet as parquet_module
from core.persistence.timeseries import timescale as timescale_module
from core.persistence.timeseries.clickhouse import ClickHouseTimeSeriesAdapter
from core.persistence.timeseries.parquet import ParquetTimeSeriesAdapter
from core.persistence.timeseries.timescale import TimescaleTimeSeriesAdapter


class _DummyCursor:
    def __init__(self, select_rows: List[Any]):
        self.queries: List[tuple[str, Any]] = []
        self._rows = select_rows

    def __enter__(self) -> "_DummyCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None

    def execute(self, sql: str, params: Any | None = None) -> None:
        self.queries.append((sql.strip(), params))

    def executemany(self, sql: str, params: Iterable[Any]) -> None:
        self.queries.append((sql.strip(), list(params)))

    def fetchall(self) -> List[Any]:
        return self._rows


class _DummyConnection:
    def __init__(self, rows: List[Any]):
        self.rows = rows
        self.cursors: List[_DummyCursor] = []

    def cursor(self) -> _DummyCursor:
        cursor = _DummyCursor(self.rows)
        self.cursors.append(cursor)
        return cursor

    def close(self) -> None:  # pragma: no cover - trivial
        return None


def test_timescale_adapter_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    select_rows = [
        (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            {"symbol": "BTCUSDT"},
            {"price": 42000.0},
        )
    ]
    connection = _DummyConnection(select_rows)
    dummy_module = SimpleNamespace(connect=lambda dsn, autocommit=True: connection)

    original_find_spec = timescale_module.importlib.util.find_spec
    original_import_module = timescale_module.importlib.import_module

    def fake_find_spec(name: str):
        if name == "psycopg":
            return object()
        return original_find_spec(name)

    def fake_import_module(name: str):
        if name == "psycopg":
            return dummy_module
        return original_import_module(name)

    monkeypatch.setattr(timescale_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(timescale_module.importlib, "import_module", fake_import_module)

    adapter = TimescaleTimeSeriesAdapter("postgresql://example")
    points = [
        TimeSeriesPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            values={"price": 42000.0},
            tags={"symbol": "BTCUSDT"},
        )
    ]

    written = adapter.write_points("vpin_ticks", points)
    assert written == 1

    rows = list(
        adapter.read_points(
            "vpin_ticks",
            start=datetime(2023, 12, 31, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )
    assert rows[0].values["price"] == 42000.0
    assert rows[0].tags == {"symbol": "BTCUSDT"}

    adapter.close()


class _DummyResult:
    def __init__(self, rows: List[Any]):
        self.result_rows = rows


class _DummyClickHouseClient:
    def __init__(self, rows: List[Any]):
        self.rows = rows
        self.ddl: List[str] = []
        self.inserts: List[Any] = []
        self.queries: List[Any] = []
        self.closed = False

    def command(self, ddl: str) -> None:
        self.ddl.append(ddl.strip())

    def insert(self, table: str, rows: List[Dict[str, Any]]) -> None:
        self.inserts.append((table, rows))

    def query(self, query: str, params: Dict[str, Any] | None = None) -> _DummyResult:
        self.queries.append((query.strip(), params))
        return _DummyResult(self.rows)

    def close(self) -> None:
        self.closed = True


def test_clickhouse_adapter_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            {"symbol": "ETHUSDT"},
            {"price": 3600.0},
        )
    ]
    client = _DummyClickHouseClient(rows)
    dummy_module = SimpleNamespace(get_client=lambda **_: client)

    original_find_spec = clickhouse_module.importlib.util.find_spec
    original_import_module = clickhouse_module.importlib.import_module

    def fake_find_spec(name: str):
        if name == "clickhouse_connect":
            return object()
        return original_find_spec(name)

    def fake_import_module(name: str):
        if name == "clickhouse_connect":
            return dummy_module
        return original_import_module(name)

    monkeypatch.setattr(clickhouse_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(clickhouse_module.importlib, "import_module", fake_import_module)

    adapter = ClickHouseTimeSeriesAdapter("localhost")
    points = [
        TimeSeriesPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            values={"price": 3600.0},
            tags={"symbol": "ETHUSDT"},
        )
    ]

    assert adapter.write_points("ohlc", points) == 1
    results = list(adapter.read_points("ohlc"))
    assert results[0].values["price"] == 3600.0
    adapter.close()
    assert client.closed is True


@dataclass
class _FakeTable:
    data: Dict[str, List[Any]]

    def to_pylist(self) -> List[Dict[str, Any]]:
        length = len(next(iter(self.data.values()))) if self.data else 0
        rows: List[Dict[str, Any]] = []
        for index in range(length):
            rows.append({key: values[index] for key, values in self.data.items()})
        return rows


class _FakeScalar:
    def __init__(self, value: Any) -> None:
        self._value = value

    def as_py(self) -> Any:
        return self._value


class _FilterExpression:
    def __init__(self, predicate) -> None:
        self._predicate = predicate

    def __call__(self, row: Dict[str, Any]) -> bool:
        return self._predicate(row)

    def __and__(self, other: "_FilterExpression") -> "_FilterExpression":
        return _FilterExpression(lambda row: self(row) and other(row))


class _FakeField:
    def __init__(self, name: str) -> None:
        self._name = name

    def __ge__(self, scalar: _FakeScalar) -> _FilterExpression:
        return _FilterExpression(lambda row: row[self._name] >= scalar.as_py())

    def __le__(self, scalar: _FakeScalar) -> _FilterExpression:
        return _FilterExpression(lambda row: row[self._name] <= scalar.as_py())


class _FakePyArrowModule:
    class Table:
        @staticmethod
        def from_pydict(data: Dict[str, List[Any]]) -> _FakeTable:
            return _FakeTable(data)

    @staticmethod
    def concat_tables(tables: List[_FakeTable]) -> _FakeTable:
        combined: Dict[str, List[Any]] = {}
        for table in tables:
            for key, values in table.data.items():
                combined.setdefault(key, []).extend(values)
        return _FakeTable(combined)

    @staticmethod
    def scalar(value: Any) -> _FakeScalar:
        return _FakeScalar(value)


class _FakeParquetModule:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]]) -> None:
        self._store = store

    def write_table(self, table: _FakeTable, path, append: bool = False) -> None:
        key = str(path)
        rows = table.to_pylist()
        if append and key in self._store:
            self._store[key].extend(rows)
        else:
            self._store[key] = rows

    def read_table(self, path) -> _FakeTable:
        rows = self._store[str(path)]
        data: Dict[str, List[Any]] = {}
        for row in rows:
            for key, value in row.items():
                data.setdefault(key, []).append(value)
        return _FakeTable(data)


class _FakeDatasetModule:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]]) -> None:
        self._store = store

    def dataset(self, path, format="parquet", partitioning="hive"):
        prefix = str(path)
        rows: List[Dict[str, Any]] = []
        for stored_path, stored_rows in self._store.items():
            if stored_path.startswith(prefix):
                rows.extend(stored_rows)

        class _Dataset:
            def __init__(self, dataset_rows: List[Dict[str, Any]]) -> None:
                self._rows = dataset_rows

            def to_table(self, filter=None, limit=None) -> _FakeTable:
                filtered = self._rows
                if filter is not None:
                    filtered = [row for row in filtered if filter(row)]
                if limit is not None:
                    filtered = filtered[:limit]
                if not filtered:
                    return _FakeTable({"timestamp": [], "tags": [], "values": []})
                data = {
                    "timestamp": [row["timestamp"] for row in filtered],
                    "tags": [row["tags"] for row in filtered],
                    "values": [row["values"] for row in filtered],
                }
                return _FakeTable(data)

        return _Dataset(rows)

    def field(self, name: str) -> _FakeField:
        return _FakeField(name)


def test_parquet_adapter_round_trip(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, List[Dict[str, Any]]] = {}

    def fake_loader(self) -> Any:
        return (
            _FakePyArrowModule(),
            _FakeParquetModule(store),
            _FakeDatasetModule(store),
        )

    monkeypatch.setattr(parquet_module.ParquetTimeSeriesAdapter, "_load_pyarrow", fake_loader, raising=False)

    adapter = ParquetTimeSeriesAdapter(tmp_path)
    timestamp = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    points = [
        TimeSeriesPoint(timestamp=timestamp, values={"price": 1.23}, tags={"symbol": "BTCUSDT"})
    ]

    assert adapter.write_points("ticks", points) == 1

    results = list(
        adapter.read_points(
            "ticks",
            start=datetime(2023, 12, 31, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )
    assert len(results) == 1
    assert results[0].values["price"] == 1.23
    assert results[0].tags["symbol"] == "BTCUSDT"


def test_parquet_adapter_gracefully_handles_missing_dataset(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store: Dict[str, List[Dict[str, Any]]] = {}

    def fake_loader(self) -> Any:
        return (
            _FakePyArrowModule(),
            _FakeParquetModule(store),
            _FakeDatasetModule(store),
        )

    monkeypatch.setattr(parquet_module.ParquetTimeSeriesAdapter, "_load_pyarrow", fake_loader, raising=False)

    adapter = ParquetTimeSeriesAdapter(tmp_path)

    assert list(adapter.read_points("absent")) == []
