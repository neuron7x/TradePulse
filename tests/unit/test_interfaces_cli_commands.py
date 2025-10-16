from __future__ import annotations

import argparse
import io
import json
from contextlib import contextmanager, redirect_stdout

from interfaces import cli


class StubMetricsCollector:
    def __init__(self) -> None:
        self.ingestion_calls: list[tuple[str, str]] = []
        self.ingestion_contexts: list[dict[str, object]] = []
        self.recorded_ticks: list[tuple[str, str, int]] = []
        self.backtest_calls: list[str] = []
        self.backtest_contexts: list[dict[str, object]] = []

    @contextmanager
    def measure_data_ingestion(self, source: str, symbol: str):
        self.ingestion_calls.append((source, symbol))
        ctx: dict[str, object] = {}
        try:
            yield ctx
        finally:
            self.ingestion_contexts.append(ctx)

    def record_tick_processed(self, source: str, symbol: str, count: int = 1) -> None:
        self.recorded_ticks.append((source, symbol, count))

    @contextmanager
    def measure_backtest(self, strategy: str):
        self.backtest_calls.append(strategy)
        ctx: dict[str, object] = {}
        try:
            yield ctx
        finally:
            self.backtest_contexts.append(ctx)


def _csv_with_duplicates(path) -> None:
    lines = [
        "ts,price,volume\n",
        "0,100,1\n",
        "1,101,1\n",
        "1,102,1\n",
        "2,103,1\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_cmd_analyze_quarantines_and_records_metrics(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "series.csv"
    _csv_with_duplicates(csv_path)
    collector = StubMetricsCollector()
    monkeypatch.setattr(cli, "get_metrics_collector", lambda registry=None: collector)

    args = argparse.Namespace(
        csv=str(csv_path),
        price_col="price",
        window=3,
        bins=10,
        delta=0.005,
        gpu=False,
        config=None,
        traceparent=None,
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli.cmd_analyze(args)

    payload = json.loads(buffer.getvalue())
    assert payload["quality"]["quarantined_rows"] == 2
    assert payload["quality"]["duplicates_rows"] == 2
    assert collector.ingestion_calls == [("csv", csv_path.stem.upper())]
    assert collector.backtest_calls == []
    assert collector.recorded_ticks == [("csv", csv_path.stem.upper(), 4)]
    assert collector.ingestion_contexts[0]["rows"] == 4
    assert payload["quality"]["quarantined_preview"]
    assert isinstance(payload["quality"]["quarantined_preview"][0]["timestamp"], str)


def test_cmd_backtest_instruments_metrics_and_returns_quality(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "backtest.csv"
    _csv_with_duplicates(csv_path)
    collector = StubMetricsCollector()
    monkeypatch.setattr(cli, "get_metrics_collector", lambda registry=None: collector)

    args = argparse.Namespace(
        csv=str(csv_path),
        price_col="price",
        window=3,
        fee=0.0005,
        gpu=False,
        config=None,
        traceparent=None,
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli.cmd_backtest(args)

    payload = json.loads(buffer.getvalue())
    assert payload["quality"]["quarantined_rows"] == 2
    assert payload["symbol"] == csv_path.stem.upper()
    assert collector.ingestion_calls == [("csv", csv_path.stem.upper())]
    assert collector.recorded_ticks == [("csv", csv_path.stem.upper(), 4)]
    assert collector.backtest_calls == ["cli.signal"]
    assert collector.backtest_contexts[0]["pnl"] == payload["pnl"]
    assert collector.backtest_contexts[0]["trades"] == payload["trades"]
