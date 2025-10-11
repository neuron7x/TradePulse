import importlib

import numpy as np
import pytest

polars_pipeline = importlib.import_module("core.data.polars_pipeline")


class DummyArrowColumn:
    def __init__(self, array: np.ndarray):
        self._array = array

    def to_numpy(self, zero_copy_only: bool = True) -> np.ndarray:  # pragma: no cover - pass through
        return self._array


class DummyArrowTable:
    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

    @property
    def num_columns(self) -> int:
        return len(self._data)

    def column(self, index: int) -> DummyArrowColumn:
        key = list(self._data.keys())[index]
        return DummyArrowColumn(self._data[key])


class DummyDataFrame:
    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

    def to_arrow(self) -> DummyArrowTable:
        return DummyArrowTable(self._data)


class DummyLazyFrame:
    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data
        self.collected = 0
        self.row_count_name: str | None = None

    def with_row_count(self, name: str) -> "DummyLazyFrame":
        self.row_count_name = name
        return self

    def collect(self, streaming: bool = True) -> DummyDataFrame:
        self.collected += 1
        return DummyDataFrame(self._data)

    def select(self, column_name: str) -> "DummyLazyFrame":
        return DummyLazyFrame({column_name: self._data[column_name]})


class StubPolars:
    class Config:
        cache_enabled = False

        @classmethod
        def set_global_string_cache(cls, enable: bool) -> None:
            cls.cache_enabled = enable

    def __init__(self, lazy_frame: DummyLazyFrame):
        self.lazy_frame = lazy_frame
        self.scan_calls: list[tuple] = []

    def scan_csv(self, *args, **kwargs) -> DummyLazyFrame:
        self.scan_calls.append((args, kwargs))
        return self.lazy_frame

    @staticmethod
    def col(name: str) -> str:
        return name


class StubArrowModule:
    def __init__(self) -> None:
        self._pool = object()
        self.calls: list[object] = []

    def default_memory_pool(self) -> object:
        return self._pool

    def set_memory_pool(self, pool: object) -> None:
        self.calls.append(pool)
        self._pool = pool


@pytest.fixture(autouse=True)
def restore_modules(monkeypatch):
    # ensure module globals reset after each test
    monkeypatch.setattr(polars_pipeline, "pl", None)
    monkeypatch.setattr(polars_pipeline, "pa", None)
    yield


def test_scan_lazy_requires_polars(monkeypatch):
    with pytest.raises(RuntimeError, match="polars is not installed"):
        polars_pipeline.scan_lazy("/tmp/example.csv")


def test_scan_lazy_uses_stub_polars(monkeypatch):
    data = {"price": np.array([1.0, 2.0, 3.0])}
    stub_lazy = DummyLazyFrame(data)
    stub_pl = StubPolars(stub_lazy)
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)

    lazy_frame = polars_pipeline.scan_lazy("/tmp/example.csv", row_count_name="row_id")

    assert lazy_frame is stub_lazy
    assert stub_lazy.row_count_name == "row_id"
    assert stub_pl.scan_calls[0][1]["columns"] is None


def test_collect_streaming_invokes_sink(monkeypatch):
    data = {"volume": np.array([10, 20])}
    stub_lazy = DummyLazyFrame(data)
    stub_pl = StubPolars(stub_lazy)
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)

    sink_calls: list[DummyDataFrame] = []

    result = polars_pipeline.collect_streaming(stub_lazy, sink=sink_calls.append)

    assert isinstance(result, DummyDataFrame)
    assert sink_calls == [result]
    assert stub_lazy.collected == 1


def test_lazy_column_zero_copy_returns_numpy_view(monkeypatch):
    data = {"returns": np.array([0.1, 0.2, 0.3])}
    stub_lazy = DummyLazyFrame(data)
    stub_pl = StubPolars(stub_lazy)
    stub_pa = StubArrowModule()
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)
    monkeypatch.setattr(polars_pipeline, "pa", stub_pa)

    array = polars_pipeline.lazy_column_zero_copy(stub_lazy, "returns")

    assert array is data["returns"]


def test_lazy_column_zero_copy_requires_arrow(monkeypatch):
    data = {"returns": np.array([0.1, 0.2, 0.3])}
    stub_lazy = DummyLazyFrame(data)
    stub_pl = StubPolars(stub_lazy)
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)

    with pytest.raises(RuntimeError, match="pyarrow is required"):
        polars_pipeline.lazy_column_zero_copy(stub_lazy, "returns")


def test_use_arrow_memory_pool_temporarily_swaps_pool(monkeypatch):
    data = {"returns": np.array([0.1, 0.2])}
    stub_lazy = DummyLazyFrame(data)
    stub_pl = StubPolars(stub_lazy)
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)
    stub_pa = StubArrowModule()
    previous_pool = stub_pa.default_memory_pool()
    replacement = object()
    monkeypatch.setattr(polars_pipeline, "pa", stub_pa)

    with polars_pipeline.use_arrow_memory_pool(replacement):
        assert stub_pa._pool is replacement

    assert stub_pa._pool is previous_pool
    assert stub_pa.calls == [replacement, previous_pool]


def test_enable_global_string_cache_delegates(monkeypatch):
    stub_lazy = DummyLazyFrame({"price": np.array([1.0])})
    stub_pl = StubPolars(stub_lazy)
    monkeypatch.setattr(polars_pipeline, "pl", stub_pl)

    polars_pipeline.enable_global_string_cache(True)

    assert stub_pl.Config.cache_enabled is True
