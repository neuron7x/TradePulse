# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
from pathlib import Path

import pytest

import core.data.ingestion as ingestion
from core.data.ingestion import DataIngestor, Ticker


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
    assert captured[0].price == pytest.approx(101.5)
    assert captured[0].volume == pytest.approx(7.0)
