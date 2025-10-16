from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import interfaces.cli as cli
import core.data.ingestion as ingestion_module


class DummyMetrics:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.records: list[tuple[str, str, int]] = []

    def measure_data_ingestion(self, source: str, symbol: str):  # type: ignore[override]
        class _Context:
            def __init__(self, outer: "DummyMetrics", src: str, sym: str) -> None:
                self._outer = outer
                self._source = src
                self._symbol = sym
                self.context: dict[str, object] = {}

            def __enter__(self) -> dict[str, object]:
                self._outer.calls.append((self._source, self._symbol, self.context))
                return self.context

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

        return _Context(self, source, symbol)

    def record_tick_processed(self, source: str, symbol: str, count: int = 1) -> None:
        self.records.append((source, symbol, count))


def _write_sample_csv(path: Path) -> None:
    rows = [
        (1.0, "100", "5"),
        (1.0, "101", "6"),
        (2.0, "102", "7"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("ts,price,volume\n")
        for ts, price, volume in rows:
            handle.write(f"{ts},{price},{volume}\n")


@pytest.fixture()
def metrics_patch(monkeypatch: pytest.MonkeyPatch) -> DummyMetrics:
    dummy = DummyMetrics()
    monkeypatch.setattr(cli, "get_metrics_collector", lambda: dummy)
    monkeypatch.setattr(ingestion_module, "get_metrics_collector", lambda: dummy)
    return dummy


def test_cmd_analyze_reports_quality_and_metrics(tmp_path: Path, capfd: pytest.CaptureFixture[str], metrics_patch: DummyMetrics) -> None:
    csv_path = tmp_path / "series.csv"
    _write_sample_csv(csv_path)
    args = Namespace(
        csv=str(csv_path),
        price_col="price",
        window=3,
        bins=10,
        delta=0.01,
        config=None,
        gpu=False,
        traceparent=None,
    )

    cli.cmd_analyze(args)
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert payload["quality"]["duplicate_rows"] == 2
    assert payload["quality"]["clean_rows"] == 2
    assert metrics_patch.calls[0][0] == "cli_csv"
    assert metrics_patch.calls[0][1] == "SERIES"
    assert metrics_patch.calls[0][2]["rows"] == 3
    assert ("cli_csv", "SERIES", 1) in metrics_patch.records


def test_cmd_backtest_includes_quality_summary(tmp_path: Path, capfd: pytest.CaptureFixture[str], metrics_patch: DummyMetrics) -> None:
    csv_path = tmp_path / "prices.csv"
    _write_sample_csv(csv_path)
    args = Namespace(
        csv=str(csv_path),
        price_col="price",
        window=3,
        fee=0.0,
        config=None,
        gpu=False,
        traceparent=None,
    )

    cli.cmd_backtest(args)
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert "quality" in payload
    assert payload["quality"]["quarantined_rows"] >= 1
    assert any(source == "cli_csv" for source, *_ in metrics_patch.records)
