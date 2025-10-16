from __future__ import annotations
import argparse
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import interfaces.cli as cli_module
from core.data.models import PriceTick
from core.data.quality_control import QualityReport
from core.indicators.entropy import delta_entropy, entropy
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.ricci import build_price_graph, mean_ricci
from core.phase.detector import composite_transition
from interfaces.cli import signal_from_indicators


def _reference_signal(prices: np.ndarray, window: int = 32, ricci_delta: float = 0.005) -> np.ndarray:
    sig = np.zeros(len(prices), dtype=int)
    for t in range(window, len(prices)):
        prefix = prices[: t + 1]
        phases = compute_phase(prefix)
        synchrony = kuramoto_order(phases[-window:])
        entropy_value = entropy(prefix[-window:])
        delta_value = delta_entropy(prefix, window=window)
        graph = build_price_graph(prefix[-window:], delta=ricci_delta)
        curvature = mean_ricci(graph)
        composite = composite_transition(synchrony, delta_value, curvature, entropy_value)
        if composite > 0.15 and delta_value < 0 and curvature < 0:
            sig[t] = 1
        elif composite < -0.15 and delta_value > 0:
            sig[t] = -1
        else:
            sig[t] = sig[t - 1]
    return sig


def test_signal_from_indicators_matches_reference() -> None:
    prices = np.linspace(100.0, 110.0, num=128)
    reference = _reference_signal(prices)

    threaded = signal_from_indicators(prices, window=32, max_workers=3)
    sequential = signal_from_indicators(prices, window=32, max_workers=0)

    assert np.array_equal(threaded, reference)
    assert np.array_equal(sequential, reference)


def test_signal_from_indicators_accepts_custom_ricci_delta() -> None:
    prices = np.linspace(100.0, 102.0, num=96)
    reference = _reference_signal(prices, window=24, ricci_delta=0.01)

    result = signal_from_indicators(prices, window=24, max_workers=2, ricci_delta=0.01)
    assert np.array_equal(result, reference)


class DummyMetrics:
    def __init__(self) -> None:
        self.measure_calls: list[tuple[str, str]] = []
        self.record_calls: list[tuple[str, str, int]] = []

    @contextmanager
    def measure_data_ingestion(self, source: str, symbol: str):
        self.measure_calls.append((source, symbol))
        yield {}

    def record_tick_processed(self, source: str, symbol: str, count: int = 1) -> None:
        self.record_calls.append((source, symbol, count))


def _quality_report(clean: pd.DataFrame) -> QualityReport:
    quarantined = clean.iloc[:1].copy()
    duplicates = clean.iloc[:1].copy()
    spikes = clean.iloc[:1].copy()
    return QualityReport(
        clean=clean,
        quarantined=quarantined,
        duplicates=duplicates,
        spikes=spikes,
    )


def _ticks() -> list[PriceTick]:
    return [
        PriceTick.create(symbol="TEST", venue="CSV", price=100 + idx, volume=1, timestamp=idx * 60)
        for idx in range(4)
    ]


def test_cmd_analyze_uses_validator_reports_quality(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ticks = _ticks()

    class DummyIngestor:
        def historical_csv(self, path, callback, **kwargs):
            for tick in ticks:
                callback(tick)

    monkeypatch.setattr(cli_module, "_make_data_ingestor", lambda _=None: DummyIngestor())

    clean = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC"),
            "price": [100.0, 101.0, 102.0, 103.0],
        }
    )
    report = _quality_report(clean)
    captured: dict[str, object] = {}

    def fake_validate(frame, config, *, price_column, **kwargs):
        captured["frame"] = frame
        captured["config"] = config
        captured["price_column"] = price_column
        return report

    monkeypatch.setattr(cli_module, "validate_and_quarantine", fake_validate)
    metrics = DummyMetrics()
    monkeypatch.setattr(cli_module, "get_metrics_collector", lambda: metrics)

    args = argparse.Namespace(
        csv="data.csv",
        price_col="price",
        window=2,
        bins=10,
        delta=0.01,
        gpu=False,
        traceparent=None,
        config=None,
    )

    cli_module.cmd_analyze(args)
    output = json.loads(capsys.readouterr().out)

    assert output["quality"]["quarantined_rows"] == report.quarantined.shape[0]
    assert output["quality"]["clean_rows"] == report.clean.shape[0]
    assert captured["price_column"] == "price"
    assert "timestamp" in captured["frame"].columns
    assert captured["config"].timestamp_column == "timestamp"
    assert metrics.measure_calls == [("csv", "data")]
    assert metrics.record_calls == [("csv", "data", report.clean.shape[0])]


def test_cmd_backtest_reports_quality_and_metrics(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    ticks = _ticks()

    class DummyIngestor:
        def historical_csv(self, path, callback, **kwargs):
            for tick in ticks:
                callback(tick)

    monkeypatch.setattr(cli_module, "_make_data_ingestor", lambda _=None: DummyIngestor())

    clean = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC"),
            "close": [200.0, 201.0, 202.0, 203.0],
        }
    )
    report = _quality_report(clean)
    captured: dict[str, object] = {}

    def fake_validate(frame, config, *, price_column, **kwargs):
        captured["frame"] = frame
        captured["price_column"] = price_column
        return report

    monkeypatch.setattr(cli_module, "validate_and_quarantine", fake_validate)
    metrics = DummyMetrics()
    monkeypatch.setattr(cli_module, "get_metrics_collector", lambda: metrics)

    captured_signal: dict[str, object] = {}

    def fake_signal(prices, window):
        captured_signal["prices"] = prices.copy()
        captured_signal["window"] = window
        return np.ones_like(prices, dtype=int)

    monkeypatch.setattr(cli_module, "signal_from_indicators", fake_signal)

    def fake_walk(prices, signal_fn, fee):
        captured_signal["walk_prices"] = prices.copy()
        captured_signal["fee"] = fee
        signals = signal_fn(prices)
        assert signals.shape == prices.shape
        return SimpleNamespace(pnl=1.0, max_dd=0.5, trades=3)

    monkeypatch.setattr(cli_module, "walk_forward", fake_walk)

    args = argparse.Namespace(
        csv="data.csv",
        price_col="close",
        window=2,
        fee=0.001,
        config=None,
        gpu=False,
        traceparent=None,
    )

    cli_module.cmd_backtest(args)
    output = json.loads(capsys.readouterr().out)

    expected_prices = clean["close"].to_numpy(dtype=float)
    np.testing.assert_allclose(captured_signal["prices"], expected_prices)
    np.testing.assert_allclose(captured_signal["walk_prices"], expected_prices)
    assert captured_signal["window"] == args.window
    assert captured_signal["fee"] == args.fee
    assert captured["price_column"] == "close"
    assert "close" in captured["frame"].columns
    assert output["quality"]["clean_rows"] == report.clean.shape[0]
    assert output["quality"]["quarantined_rows"] == report.quarantined.shape[0]
    assert metrics.measure_calls == [("csv", "data")]
    assert metrics.record_calls == [("csv", "data", report.clean.shape[0])]
