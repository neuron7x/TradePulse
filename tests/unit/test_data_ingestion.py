# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path

import pytest

import core.data.ingestion as ingestion
from core.data.ingestion import BinanceStreamHandle, DataIngestor, Ticker


class DummyMetricsCollector:
    def __init__(self) -> None:
        self.measure_calls: list[dict[str, object]] = []
        self.tick_calls: list[dict[str, object]] = []

    @contextmanager
    def measure_data_ingestion(self, source: str, symbol: str):  # type: ignore[no-untyped-def]
        ctx: dict[str, object] = {}
        self.measure_calls.append({"source": source, "symbol": symbol, "ctx": ctx})
        yield ctx

    def record_tick_processed(self, source: str, symbol: str, count: int = 1) -> None:
        self.tick_calls.append({"source": source, "symbol": symbol, "count": count})


def test_historical_csv_reads_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "price", "volume"])
        writer.writeheader()
        writer.writerow({"ts": "1", "price": "100", "volume": "5"})
        writer.writerow({"ts": "2", "price": "101", "volume": "6"})
    ingestor = DataIngestor()
    records: list[Ticker] = []
    ingestor.historical_csv(str(csv_path), records.append)
    assert len(records) == 2
    assert records[0].price == 100.0
    assert records[1].volume == 6.0


def test_historical_csv_records_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "price", "volume"])
        writer.writeheader()
        writer.writerow({"ts": "1", "price": "100", "volume": "5"})
        writer.writerow({"ts": "2", "price": "101", "volume": "6"})

    metrics = DummyMetricsCollector()
    monkeypatch.setattr(ingestion, "_METRICS_COLLECTOR", metrics)

    ingestor = DataIngestor()
    ingestor.historical_csv(str(csv_path), lambda _: None, symbol="BTCUSDT", venue="TEST")

    assert metrics.measure_calls
    call = metrics.measure_calls[0]
    assert call["source"] == "csv"
    assert call["symbol"] == "BTCUSDT"
    ctx = call["ctx"]
    assert ctx["venue"] == "TEST"
    assert ctx["path"] == str(csv_path)
    assert ctx["instrument_type"] == "spot"
    assert metrics.tick_calls == [
        {"source": "csv", "symbol": "BTCUSDT", "count": 1},
        {"source": "csv", "symbol": "BTCUSDT", "count": 1},
    ]


def test_binance_ws_requires_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = DataIngestor()
    with pytest.raises(RuntimeError):
        ingestor.binance_ws("BTCUSDT", lambda _: None)


def test_binance_ws_emits_ticks_when_dependency_available(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Ticker] = []

    class DummyWS:
        def start(self) -> None:
            pass

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            callback({"e": "kline", "k": {"T": 2000, "c": "101.5", "v": "7.0"}})

    monkeypatch.setattr(ingestion, "BinanceWS", DummyWS)
    ingestor = DataIngestor()
    ws = ingestor.binance_ws("BTCUSDT", captured.append)
    assert isinstance(ws, DummyWS)
    assert float(captured[0].price) == pytest.approx(101.5)
    assert float(captured[0].volume) == pytest.approx(7.0)


def test_binance_ws_records_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = DummyMetricsCollector()
    monkeypatch.setattr(ingestion, "_METRICS_COLLECTOR", metrics)

    captured: list[Ticker] = []

    class DummyWS:
        def start(self) -> None:
            pass

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            callback({"e": "kline", "k": {"T": 2000, "c": "101.5", "v": "7.0"}})

    monkeypatch.setattr(ingestion, "BinanceWS", DummyWS)

    ingestor = DataIngestor()
    ws = ingestor.binance_ws("BTCUSDT", captured.append, interval="1m")

    assert isinstance(ws, DummyWS)
    assert metrics.measure_calls
    call = metrics.measure_calls[0]
    assert call["source"] == "binance_ws"
    assert call["symbol"] == "BTCUSDT"
    ctx = call["ctx"]
    assert ctx["venue"] == "BINANCE"
    assert ctx["interval"] == "1m"
    assert metrics.tick_calls == [
        {"source": "binance_ws", "symbol": "BTCUSDT", "count": 1}
    ]


def test_binance_stream_handle_manages_lifecycle() -> None:
    events: list[str] = []

    class DummyWS:
        def __init__(self) -> None:
            self.started = 0
            self.stopped = 0
            self.kline_args: tuple[str, int, str, object] | None = None

        def start(self) -> None:
            self.started += 1
            events.append("start")

        def stop(self) -> None:
            self.stopped += 1
            events.append("stop")

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            self.kline_args = (symbol, id, interval, callback)

    ws = DummyWS()
    handle = BinanceStreamHandle(ws)

    callback_invocations: list[object] = []
    handle.start(symbol="BTCUSDT", interval="1m", callback=callback_invocations.append)

    assert ws.started == 1
    assert ws.kline_args is not None
    symbol, subscription_id, interval, callback = ws.kline_args
    assert symbol == "btcusdt"  # symbol is lowered for the websocket subscription
    assert subscription_id == 1
    assert interval == "1m"

    handle.close()
    assert ws.stopped == 1
    handle.close()  # should be a no-op when already closed
    assert ws.stopped == 1
    assert events == ["start", "stop"]


def test_binance_stream_handle_context_manager_closes_active_stream() -> None:
    class DummyWS:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.started = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stop_calls += 1

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            self.callback = callback

    ws = DummyWS()
    handle = BinanceStreamHandle(ws)

    with handle as active_handle:
        assert active_handle is handle
        handle.start(symbol="ETHUSDT", interval="5m", callback=lambda _: None)

    assert ws.started == 1
    assert ws.stop_calls == 1


def test_binance_ws_ignores_messages_without_kline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Ticker] = []

    class DummyWS:
        def __init__(self) -> None:
            self.kline_callback = None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            self.kline_callback = callback

    monkeypatch.setattr(ingestion, "BinanceWS", DummyWS)
    ingestor = DataIngestor()
    ws = ingestor.binance_ws("BTCUSDT", captured.append)
    assert isinstance(ws, DummyWS)
    assert ws.kline_callback is not None

    ws.kline_callback({"e": "kline"})  # type: ignore[operator]
    assert captured == []

    # ensure the attached stream handle can be closed without errors
    assert hasattr(ws, "stream_handle")
    ws.stream_handle.close()  # type: ignore[attr-defined]


def test_binance_ws_logs_warning_on_invalid_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    captured: list[Ticker] = []

    class DummyWS:
        def __init__(self) -> None:
            self.kline_callback = None

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            self.kline_callback = callback

    monkeypatch.setattr(ingestion, "BinanceWS", DummyWS)
    ingestor = DataIngestor()
    ws = ingestor.binance_ws("BTCUSDT", captured.append)
    assert isinstance(ws, DummyWS)
    assert ws.kline_callback is not None

    with caplog.at_level("WARNING"):
        ws.kline_callback({"k": {"T": "bad-ts", "c": "not-a-number"}})  # type: ignore[operator]

    assert captured == []
    assert any("Failed to parse websocket payload" in message for message in caplog.messages)
    ws.stream_handle.close()  # type: ignore[attr-defined]


def test_binance_ws_metrics_error_sets_status(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = DummyMetricsCollector()
    monkeypatch.setattr(ingestion, "_METRICS_COLLECTOR", metrics)

    class DummyWS:
        def start(self) -> None:
            raise RuntimeError("boom")

        def kline(self, symbol, id, interval, callback):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(ingestion, "BinanceWS", DummyWS)
    ingestor = DataIngestor()

    with pytest.raises(RuntimeError, match="boom"):
        ingestor.binance_ws("BTCUSDT", lambda _: None)

    assert metrics.measure_calls
    ctx = metrics.measure_calls[0]["ctx"]
    assert ctx.get("status") == "error"


def test_historical_csv_requires_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "history_no_header.csv"
    csv_path.write_text("", encoding="utf-8")
    ingestor = DataIngestor()

    with pytest.raises(ValueError, match="header"):
        ingestor.historical_csv(str(csv_path), lambda _: None)


def test_historical_csv_metrics_error_sets_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    csv_path = tmp_path / "history_no_header.csv"
    csv_path.write_text("", encoding="utf-8")

    metrics = DummyMetricsCollector()
    monkeypatch.setattr(ingestion, "_METRICS_COLLECTOR", metrics)

    ingestor = DataIngestor()

    with pytest.raises(ValueError):
        ingestor.historical_csv(str(csv_path), lambda _: None, symbol="ETHUSDT")

    assert metrics.measure_calls
    ctx = metrics.measure_calls[0]["ctx"]
    assert ctx.get("status") == "error"


def test_historical_csv_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "history_missing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "volume"])
        writer.writeheader()
        writer.writerow({"ts": "1", "volume": "5"})

    ingestor = DataIngestor()

    with pytest.raises(ValueError, match="missing required columns"):
        ingestor.historical_csv(str(csv_path), lambda _: None)


def test_historical_csv_skips_malformed_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "history_malformed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "price", "volume"])
        writer.writeheader()
        writer.writerow({"ts": "1", "price": "invalid", "volume": "5"})
        writer.writerow({"ts": "2", "price": "101", "volume": "6"})

    ingestor = DataIngestor()
    collected: list[Ticker] = []

    with caplog.at_level("WARNING"):
        ingestor.historical_csv(str(csv_path), collected.append)

    assert len(collected) == 1
    assert collected[0].price == pytest.approx(101.0)
    assert any("Skipping malformed row" in message for message in caplog.messages)


def test_historical_csv_rejects_path_outside_allowlist(tmp_path: Path) -> None:
    csv_path = tmp_path / "outside.csv"
    csv_path.write_text("ts,price\n1,1\n", encoding="utf-8")
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    ingestor = DataIngestor(allowed_roots=[allowed_root])

    with pytest.raises(PermissionError):
        ingestor.historical_csv(str(csv_path), lambda _: None)


def test_historical_csv_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    target.write_text("ts,price\n1,1\n", encoding="utf-8")
    symlink = tmp_path / "link.csv"
    try:
        symlink.symlink_to(target)
    except OSError:  # pragma: no cover - platform without symlink support
        pytest.skip("symlinks not supported")

    ingestor = DataIngestor(allowed_roots=[tmp_path])

    with pytest.raises(PermissionError):
        ingestor.historical_csv(str(symlink), lambda _: None)


def test_historical_csv_enforces_size_limit(tmp_path: Path) -> None:
    csv_path = tmp_path / "large.csv"
    csv_path.write_text("ts,price\n" + "\n".join("1,1" for _ in range(40)), encoding="utf-8")

    ingestor = DataIngestor(allowed_roots=[tmp_path], max_csv_bytes=32)

    with pytest.raises(ValueError, match="exceeds"):
        ingestor.historical_csv(str(csv_path), lambda _: None)
