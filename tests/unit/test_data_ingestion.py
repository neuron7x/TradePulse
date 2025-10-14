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


def test_binance_ws_emits_ticks_when_dependency_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_historical_csv_requires_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "history_no_header.csv"
    csv_path.write_text("", encoding="utf-8")
    ingestor = DataIngestor()

    with pytest.raises(ValueError, match="header"):
        ingestor.historical_csv(str(csv_path), lambda _: None)


def test_historical_csv_validates_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "history_missing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "volume"])
        writer.writeheader()
        writer.writerow({"ts": "1", "volume": "5"})

    ingestor = DataIngestor()

    with pytest.raises(ValueError, match="missing required columns"):
        ingestor.historical_csv(str(csv_path), lambda _: None)


def test_historical_csv_skips_malformed_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
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
    csv_path.write_text(
        "ts,price\n" + "\n".join("1,1" for _ in range(40)), encoding="utf-8"
    )

    ingestor = DataIngestor(allowed_roots=[tmp_path], max_csv_bytes=32)

    with pytest.raises(ValueError, match="exceeds"):
        ingestor.historical_csv(str(csv_path), lambda _: None)
