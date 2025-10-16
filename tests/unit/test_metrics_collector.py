from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from core.utils.metrics import MetricsCollector


def _sample_value(registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None) -> float | None:
    """Helper to extract metric samples from the registry."""

    return registry.get_sample_value(name, labels or {})


def _p2_quantile(samples: Iterable[float], probability: float) -> float:
    """Approximate quantile using the P² algorithm for benchmarking."""

    values = list(map(float, samples))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    if len(values) < 5:
        if not values:
            raise ValueError("P² requires at least one sample")
        ordered = sorted(values)
        pos = probability * (len(ordered) - 1)
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return ordered[lower]
        weight = pos - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    ordered = sorted(values[:5])
    positions = [1, 2, 3, 4, 5]
    desired = [1.0, 1.0 + 2 * probability, 1.0 + 4 * probability, 3.0 + 2 * probability, 5.0]
    increments = [0.0, probability / 2.0, probability, (1.0 + probability) / 2.0, 1.0]

    for x in values[5:]:
        if x < ordered[0]:
            ordered[0] = x
            k = 0
        elif x < ordered[1]:
            k = 0
        elif x < ordered[2]:
            k = 1
        elif x < ordered[3]:
            k = 2
        elif x < ordered[4]:
            k = 3
        else:
            ordered[4] = x
            k = 3

        for i in range(k + 1, 5):
            positions[i] += 1

        for i in range(5):
            desired[i] += increments[i]

        for i in range(1, 4):
            d = desired[i] - positions[i]
            if (d >= 1 and positions[i + 1] - positions[i] > 1) or (d <= -1 and positions[i - 1] - positions[i] < -1):
                d_sign = 1 if d > 0 else -1
                new_height = ordered[i] + d_sign / (positions[i + 1] - positions[i - 1]) * (
                    (positions[i] - positions[i - 1] + d_sign) * (ordered[i + 1] - ordered[i]) / (positions[i + 1] - positions[i])
                    + (positions[i + 1] - positions[i] - d_sign) * (ordered[i] - ordered[i - 1]) / (positions[i] - positions[i - 1])
                )
                if ordered[i - 1] < new_height < ordered[i + 1]:
                    ordered[i] = new_height
                else:
                    if d_sign > 0:
                        ordered[i] += (ordered[i + 1] - ordered[i]) / (positions[i + 1] - positions[i])
                    else:
                        ordered[i] += (ordered[i - 1] - ordered[i]) / (positions[i - 1] - positions[i])
                positions[i] += d_sign

    return float(ordered[2])


def test_measure_data_ingestion_records_duration_and_status() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with collector.measure_data_ingestion("csv", "BTC-USDT"):
        pass

    count = _sample_value(
        registry,
        "tradepulse_data_ingestion_duration_seconds_count",
        {"source": "csv", "symbol": "BTC-USDT"},
    )
    total = _sample_value(
        registry,
        "tradepulse_data_ingestion_total",
        {"source": "csv", "symbol": "BTC-USDT", "status": "success"},
    )
    latency_quantile = _sample_value(
        registry,
        "tradepulse_data_ingestion_latency_quantiles_seconds",
        {"source": "csv", "symbol": "BTC-USDT", "quantile": "p50"},
    )

    assert count == 1.0
    assert total == 1.0
    assert latency_quantile is not None


def test_order_placement_context_uses_custom_status_and_updates_gauges() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with collector.measure_order_placement("binance", "ETH-USDT", "limit") as ctx:
        ctx["status"] = "rejected"

    collector.set_open_positions("binance", "ETH-USDT", 2)
    collector.set_strategy_score("alpha", 0.87)
    collector.set_strategy_memory_size(4)
    collector.record_tick_processed("csv", "ETH-USDT", count=3)
    collector.record_order_placed("binance", "ETH-USDT", "market", status="success", count=2)

    duration_count = _sample_value(
        registry,
        "tradepulse_order_placement_duration_seconds_count",
        {"exchange": "binance", "symbol": "ETH-USDT"},
    )
    rejected_total = _sample_value(
        registry,
        "tradepulse_orders_placed_total",
        {"exchange": "binance", "symbol": "ETH-USDT", "order_type": "limit", "status": "rejected"},
    )
    open_positions = _sample_value(
        registry,
        "tradepulse_open_positions",
        {"exchange": "binance", "symbol": "ETH-USDT"},
    )
    strategy_score = _sample_value(
        registry,
        "tradepulse_strategy_score",
        {"strategy_name": "alpha"},
    )
    memory_size = _sample_value(registry, "tradepulse_strategy_memory_size")
    ticks_processed = _sample_value(
        registry,
        "tradepulse_ticks_processed_total",
        {"source": "csv", "symbol": "ETH-USDT"},
    )
    market_orders = _sample_value(
        registry,
        "tradepulse_orders_placed_total",
        {"exchange": "binance", "symbol": "ETH-USDT", "order_type": "market", "status": "success"},
    )

    assert duration_count == 1.0
    submission_quantile = _sample_value(
        registry,
        "tradepulse_order_submission_latency_quantiles_seconds",
        {"exchange": "binance", "symbol": "ETH-USDT", "quantile": "p50"},
    )
    assert submission_quantile is not None
    assert rejected_total == 1.0
    assert open_positions == 2.0
    assert strategy_score == 0.87
    assert memory_size == 4.0
    assert ticks_processed == 3.0
    assert market_orders == 2.0


def test_trade_latency_and_slippage_helpers_record_samples() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    collector.observe_trade_latency_ms("binance", "RESTWebSocketConnector", "BTCUSDT", "limit", 125.0)
    collector.observe_trade_latency_ms("binance", "RESTWebSocketConnector", "BTCUSDT", "limit", 75.0)
    collector.observe_slippage_bps("binance", "BTCUSDT", "buy", 12.5)
    collector.observe_slippage_bps("binance", "BTCUSDT", "buy", -5.0)

    latency_count = _sample_value(
        registry,
        "trade_latency_ms_count",
        {"exchange": "binance", "adapter": "RESTWebSocketConnector", "symbol": "BTCUSDT", "order_type": "limit"},
    )
    latency_sum = _sample_value(
        registry,
        "trade_latency_ms_sum",
        {"exchange": "binance", "adapter": "RESTWebSocketConnector", "symbol": "BTCUSDT", "order_type": "limit"},
    )
    slippage_count = _sample_value(
        registry,
        "slippage_bps_count",
        {"exchange": "binance", "symbol": "BTCUSDT", "side": "buy"},
    )
    adverse_bucket = _sample_value(
        registry,
        "slippage_bps_bucket",
        {"exchange": "binance", "symbol": "BTCUSDT", "side": "buy", "le": "25.0"},
    )
    favorable_bucket = _sample_value(
        registry,
        "slippage_bps_bucket",
        {"exchange": "binance", "symbol": "BTCUSDT", "side": "buy", "le": "0.0"},
    )

    assert latency_count == 2.0
    assert latency_sum == pytest.approx(200.0)
    assert slippage_count == 2.0
    assert adverse_bucket == 2.0
    assert favorable_bucket == 1.0


def test_render_prometheus_disabled_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr("core.utils.metrics.PROMETHEUS_AVAILABLE", False)
    collector = MetricsCollector()

    assert collector.render_prometheus() == ""


def test_data_ingestion_context_ignores_none_and_blank_status_overrides() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with collector.measure_data_ingestion("api", "BTC-USDT") as ctx:
        ctx["status"] = None

    success_total = _sample_value(
        registry,
        "tradepulse_data_ingestion_total",
        {"source": "api", "symbol": "BTC-USDT", "status": "success"},
    )
    none_total = _sample_value(
        registry,
        "tradepulse_data_ingestion_total",
        {"source": "api", "symbol": "BTC-USDT", "status": "None"},
    )

    assert success_total == 1.0
    assert none_total is None

    second_registry = CollectorRegistry()
    collector = MetricsCollector(second_registry)

    with collector.measure_data_ingestion("api", "ETH-USDT") as ctx:
        ctx["status"] = "   "

    blank_success = _sample_value(
        second_registry,
        "tradepulse_data_ingestion_total",
        {"source": "api", "symbol": "ETH-USDT", "status": "success"},
    )
    blank_override = _sample_value(
        second_registry,
        "tradepulse_data_ingestion_total",
        {"source": "api", "symbol": "ETH-USDT", "status": ""},
    )

    assert blank_success == 1.0
    assert blank_override is None


def test_order_placement_context_forces_error_status_on_exception() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with pytest.raises(RuntimeError):
        with collector.measure_order_placement("binance", "BTC-USDT", "limit") as ctx:
            ctx["status"] = "filled"
            raise RuntimeError("order placement failed")

    error_total = _sample_value(
        registry,
        "tradepulse_orders_placed_total",
        {
            "exchange": "binance",
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "status": "error",
        },
    )
    filled_total = _sample_value(
        registry,
        "tradepulse_orders_placed_total",
        {
            "exchange": "binance",
            "symbol": "BTC-USDT",
            "order_type": "limit",
            "status": "filled",
        },
    )

    assert error_total == 1.0
    assert filled_total is None


def test_latency_quantiles_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry()

    # Simulate an environment where numpy is unavailable so the fallback code path is used.
    monkeypatch.setattr("core.utils.metrics._NUMPY_AVAILABLE", False, raising=False)
    monkeypatch.setattr("core.utils.metrics.np", None, raising=False)

    # Provide deterministic timing so the computed duration is stable.
    times = [0.0, 0.25]

    def fake_time() -> float:
        return times.pop(0) if times else 0.25

    monkeypatch.setattr("core.utils.metrics.time.time", fake_time)

    collector = MetricsCollector(registry)

    with collector.measure_data_ingestion("csv", "BTC-USDT"):
        pass

    quantile = _sample_value(
        registry,
        "tradepulse_data_ingestion_latency_quantiles_seconds",
        {"source": "csv", "symbol": "BTC-USDT", "quantile": "p95"},
    )

    # With deterministic timing and the fallback path, the quantile should match the
    # observed duration, demonstrating that the computation works without numpy.
    assert quantile == pytest.approx(0.25)


def test_signal_generation_latency_and_equity_curve_gauge() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with collector.measure_signal_generation("trend"):
        pass

    collector.record_equity_point("trend", 0, 100.0)
    collector.record_order_fill_latency("demo", "trend", 0.123)
    collector.record_order_ack_latency("demo", "trend", 0.045)
    collector.record_signal_to_fill_latency("trend", "demo", "trend", 0.25)

    signal_total = _sample_value(
        registry,
        "tradepulse_signal_generation_total",
        {"strategy": "trend", "status": "success"},
    )
    signal_quantile = _sample_value(
        registry,
        "tradepulse_signal_generation_latency_quantiles_seconds",
        {"strategy": "trend", "quantile": "p50"},
    )
    fill_quantile = _sample_value(
        registry,
        "tradepulse_order_fill_latency_quantiles_seconds",
        {"exchange": "demo", "symbol": "trend", "quantile": "p50"},
    )
    ack_quantile = _sample_value(
        registry,
        "tradepulse_order_ack_latency_quantiles_seconds",
        {"exchange": "demo", "symbol": "trend", "quantile": "p50"},
    )
    signal_fill_quantile = _sample_value(
        registry,
        "tradepulse_signal_to_fill_latency_quantiles_seconds",
        {"strategy": "trend", "exchange": "demo", "symbol": "trend", "quantile": "p50"},
    )
    equity_gauge = _sample_value(
        registry,
        "tradepulse_backtest_equity_curve",
        {"strategy": "trend", "step": "0"},
    )

    assert signal_total == 1.0
    assert signal_quantile is not None
    assert fill_quantile is not None
    assert ack_quantile is not None
    assert signal_fill_quantile is not None
    assert equity_gauge == 100.0


@pytest.mark.parametrize(
    "gauge_name,labels",
    [
        ("data_ingestion_latency_quantiles", {"source": "csv", "symbol": "BTC-USDT"}),
        ("signal_generation_latency_quantiles", {"strategy": "trend"}),
        ("order_submission_latency_quantiles", {"exchange": "demo", "symbol": "ETH-USDT"}),
        ("order_fill_latency_quantiles", {"exchange": "demo", "symbol": "ETH-USDT"}),
        ("order_ack_latency_quantiles", {"exchange": "demo", "symbol": "ETH-USDT"}),
        (
            "signal_to_fill_latency_quantiles",
            {"strategy": "trend", "exchange": "demo", "symbol": "ETH-USDT"},
        ),
    ],
)
def test_deterministic_quantiles_match_numpy_and_benchmark(gauge_name: str, labels: dict[str, str]) -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    durations = list(np.linspace(0.01, 1.0, 200))
    samples = deque(durations, maxlen=256)

    gauge = getattr(collector, gauge_name)
    collector._update_latency_quantiles(gauge, labels, samples)

    for quantile, suffix in zip((0.5, 0.95, 0.99), ("p50", "p95", "p99")):
        observed = _sample_value(
            registry,
            gauge._name,  # type: ignore[attr-defined]
            {**labels, "quantile": suffix},
        )
        assert observed is not None
        expected = np.quantile(durations, quantile)
        np.testing.assert_allclose(observed, expected)

        p2_estimate = _p2_quantile(durations, quantile)
        assert abs(p2_estimate - observed) <= max(0.05, 0.05 * observed)


def test_record_regression_metrics_sets_gauges() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    collector.record_regression_metrics("kuramoto", mae=0.12, rmse=0.3, r2=0.85)

    mae_value = _sample_value(
        registry,
        "tradepulse_regression_metric",
        {"model": "kuramoto", "metric": "mae"},
    )
    rmse_value = _sample_value(
        registry,
        "tradepulse_regression_metric",
        {"model": "kuramoto", "metric": "rmse"},
    )
    r2_value = _sample_value(
        registry,
        "tradepulse_regression_metric",
        {"model": "kuramoto", "metric": "r2"},
    )

    assert mae_value == pytest.approx(0.12)
    assert rmse_value == pytest.approx(0.3)
    assert r2_value == pytest.approx(0.85)
