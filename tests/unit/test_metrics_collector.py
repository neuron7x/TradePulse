from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from core.utils.metrics import MetricsCollector


def _sample_value(registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None) -> float | None:
    """Helper to extract metric samples from the registry."""

    return registry.get_sample_value(name, labels or {})


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


def test_signal_generation_latency_and_equity_curve_gauge() -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)

    with collector.measure_signal_generation("trend"):
        pass

    collector.record_equity_point("trend", 0, 100.0)
    collector.record_order_fill_latency("demo", "trend", 0.123)

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
    equity_gauge = _sample_value(
        registry,
        "tradepulse_backtest_equity_curve",
        {"strategy": "trend", "step": "0"},
    )

    assert signal_total == 1.0
    assert signal_quantile is not None
    assert fill_quantile is not None
    assert equity_gauge == 100.0
