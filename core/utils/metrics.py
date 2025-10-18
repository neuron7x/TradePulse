# SPDX-License-Identifier: MIT
"""Prometheus metrics collection for TradePulse.

This module provides instrumentation for all critical entrypoints and
performance-sensitive operations.
"""
from __future__ import annotations

import math
import multiprocessing
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence

try:  # pragma: no cover - exercised indirectly in environments without numpy
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled in fallback logic
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False
    _accelerated_quantiles = None
else:  # pragma: no cover - covered via normal test environment
    _NUMPY_AVAILABLE = True
    from core.accelerators.numeric import quantiles as _accelerated_quantiles

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        start_http_server,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def _fallback_quantiles(
    values: list[float], quantiles: tuple[float, ...]
) -> Dict[float, float]:
    """Compute quantiles without numpy."""

    if not values:
        return {}

    sorted_values = sorted(values)
    n = len(sorted_values)
    results: Dict[float, float] = {}

    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            continue

        position = q * (n - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        lower = sorted_values[lower_index]
        upper = sorted_values[upper_index]

        if lower_index == upper_index:
            results[q] = float(lower)
            continue

        weight = position - lower_index
        results[q] = float(lower + (upper - lower) * weight)

    return results


if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type hints
    from analytics.environment_parity import MetricDeviation


class MetricsCollector:
    """Centralized metrics collection for TradePulse."""

    def __init__(self, registry: Optional[Any] = None):
        """Initialize metrics collector.

        Args:
            registry: Prometheus registry (uses default if None)
        """
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            self.registry = None
            return

        self._enabled = True
        self.registry = registry

        # Feature/Indicator metrics
        self.feature_transform_duration = Histogram(
            "tradepulse_feature_transform_duration_seconds",
            "Time spent computing feature transformations",
            ["feature_name", "feature_type"],
            registry=registry,
        )

        self.feature_transform_total = Counter(
            "tradepulse_feature_transform_total",
            "Total number of feature transformations",
            ["feature_name", "feature_type", "status"],
            registry=registry,
        )

        self.feature_value = Gauge(
            "tradepulse_feature_value",
            "Current feature value",
            ["feature_name"],
            registry=registry,
        )

        # Backtest metrics
        self.backtest_duration = Histogram(
            "tradepulse_backtest_duration_seconds",
            "Time spent running backtests",
            ["strategy"],
            registry=registry,
        )

        self.backtest_total = Counter(
            "tradepulse_backtest_total",
            "Total number of backtests run",
            ["strategy", "status"],
            registry=registry,
        )

        self.backtest_pnl = Gauge(
            "tradepulse_backtest_pnl",
            "Backtest profit and loss",
            ["strategy"],
            registry=registry,
        )

        self.backtest_max_drawdown = Gauge(
            "tradepulse_backtest_max_drawdown",
            "Backtest maximum drawdown",
            ["strategy"],
            registry=registry,
        )

        self.backtest_trades = Gauge(
            "tradepulse_backtest_trades",
            "Number of trades in backtest",
            ["strategy"],
            registry=registry,
        )

        # Environment parity metrics
        self.environment_parity_checks = Counter(
            "tradepulse_environment_parity_checks_total",
            "Total number of environment parity evaluations grouped by status",
            ["strategy", "status"],
            registry=registry,
        )

        self.environment_parity_metric_deviation = Gauge(
            "tradepulse_environment_parity_metric_deviation",
            "Absolute deviation observed between environment metric pairs",
            ["strategy", "metric", "baseline", "comparison"],
            registry=registry,
        )

        # Data ingestion metrics
        self.data_ingestion_duration = Histogram(
            "tradepulse_data_ingestion_duration_seconds",
            "Time spent ingesting data",
            ["source", "symbol"],
            registry=registry,
        )

        self.data_ingestion_total = Counter(
            "tradepulse_data_ingestion_total",
            "Total number of data ingestion operations",
            ["source", "symbol", "status"],
            registry=registry,
        )

        self.data_ingestion_latency_quantiles = Gauge(
            "tradepulse_data_ingestion_latency_quantiles_seconds",
            "Data ingestion latency quantiles",
            ["source", "symbol", "quantile"],
            registry=registry,
        )

        self.data_ingestion_throughput = Gauge(
            "tradepulse_data_ingestion_throughput_ticks_per_second",
            "Instantaneous ingestion throughput expressed as ticks per second",
            ["source", "symbol"],
            registry=registry,
        )

        self.ticks_processed = Counter(
            "tradepulse_ticks_processed_total",
            "Total number of ticks processed",
            ["source", "symbol"],
            registry=registry,
        )

        # Watchdog metrics
        self.watchdog_worker_restarts = Counter(
            "tradepulse_watchdog_worker_restarts_total",
            "Total number of worker restarts triggered by watchdog supervision",
            ["watchdog", "worker"],
            registry=registry,
        )

        self.watchdog_live_probe_status = Gauge(
            "tradepulse_watchdog_live_probe_status",
            "Outcome of the most recent watchdog live probe (1=healthy, 0=unhealthy)",
            ["watchdog"],
            registry=registry,
        )

        self.watchdog_last_heartbeat = Gauge(
            "tradepulse_watchdog_last_heartbeat_timestamp",
            "Unix timestamp of the last watchdog heartbeat publish",
            ["watchdog"],
            registry=registry,
        )

        # Execution metrics
        self.order_placement_duration = Histogram(
            "tradepulse_order_placement_duration_seconds",
            "Time spent placing orders",
            ["exchange", "symbol"],
            registry=registry,
        )

        self.orders_placed = Counter(
            "tradepulse_orders_placed_total",
            "Total number of orders placed",
            ["exchange", "symbol", "order_type", "status"],
            registry=registry,
        )

        self.order_submission_latency_quantiles = Gauge(
            "tradepulse_order_submission_latency_quantiles_seconds",
            "Order submission latency quantiles",
            ["exchange", "symbol", "quantile"],
            registry=registry,
        )

        self.order_ack_latency_quantiles = Gauge(
            "tradepulse_order_ack_latency_quantiles_seconds",
            "Latency between order submission and broker acknowledgement",
            ["exchange", "symbol", "quantile"],
            registry=registry,
        )

        self.risk_validation_total = Counter(
            "tradepulse_risk_validations_total",
            "Total risk validation outcomes",
            ["symbol", "outcome"],
            registry=registry,
        )

        self.kill_switch_triggers_total = Counter(
            "tradepulse_kill_switch_triggers_total",
            "Kill switch triggers grouped by reason",
            ["reason"],
            registry=registry,
        )

        self.compliance_checks_total = Counter(
            "tradepulse_compliance_checks_total",
            "Compliance check outcomes",
            ["symbol", "status"],
            registry=registry,
        )

        self.compliance_violations_total = Counter(
            "tradepulse_compliance_violations_total",
            "Compliance violations by type",
            ["symbol", "violation_type"],
            registry=registry,
        )

        self.order_fill_latency_quantiles = Gauge(
            "tradepulse_order_fill_latency_quantiles_seconds",
            "Order fill latency quantiles",
            ["exchange", "symbol", "quantile"],
            registry=registry,
        )

        self.signal_to_fill_latency_quantiles = Gauge(
            "tradepulse_signal_to_fill_latency_quantiles_seconds",
            "Aggregate latency from signal emission to final fill",
            ["strategy", "exchange", "symbol", "quantile"],
            registry=registry,
        )

        self.open_positions = Gauge(
            "tradepulse_open_positions",
            "Number of open positions",
            ["exchange", "symbol"],
            registry=registry,
        )

        # Strategy metrics
        self.strategy_score = Gauge(
            "tradepulse_strategy_score",
            "Strategy performance score",
            ["strategy_name"],
            registry=registry,
        )

        self.strategy_memory_size = Gauge(
            "tradepulse_strategy_memory_size",
            "Number of strategies in memory",
            registry=registry,
        )

        self.backtest_equity_curve = Gauge(
            "tradepulse_backtest_equity_curve",
            "Equity curve samples for backtests",
            ["strategy", "step"],
            registry=registry,
        )

        self.regression_metrics = Gauge(
            "tradepulse_regression_metric",
            "Regression quality metrics (e.g. MAE, RMSE, R2)",
            ["model", "metric"],
            registry=registry,
        )

        self.signal_generation_latency_quantiles = Gauge(
            "tradepulse_signal_generation_latency_quantiles_seconds",
            "Signal generation latency quantiles",
            ["strategy", "quantile"],
            registry=registry,
        )

        self.signal_generation_total = Counter(
            "tradepulse_signal_generation_total",
            "Total number of signal generation calls",
            ["strategy", "status"],
            registry=registry,
        )

        self.health_check_latency = Histogram(
            "tradepulse_health_check_latency_seconds",
            "Latency of periodic system health probes",
            ["check_name"],
            registry=registry,
        )

        self.health_check_status = Gauge(
            "tradepulse_health_check_status",
            "Outcome of the latest health probe (1=healthy, 0=unhealthy)",
            ["check_name"],
            registry=registry,
        )

        self._ingestion_latency_samples: Dict[tuple[str, str], deque[float]] = (
            defaultdict(lambda: deque(maxlen=256))
        )
        self._signal_latency_samples: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=256)
        )
        self._order_submission_latency_samples: Dict[tuple[str, str], deque[float]] = (
            defaultdict(lambda: deque(maxlen=256))
        )
        self._order_ack_latency_samples: Dict[tuple[str, str], deque[float]] = (
            defaultdict(lambda: deque(maxlen=256))
        )
        self._order_fill_latency_samples: Dict[tuple[str, str], deque[float]] = (
            defaultdict(lambda: deque(maxlen=256))
        )
        self._signal_to_fill_latency_samples: Dict[
            tuple[str, str, str], deque[float]
        ] = defaultdict(lambda: deque(maxlen=256))

        # Agent/optimization metrics
        self.optimization_duration = Histogram(
            "tradepulse_optimization_duration_seconds",
            "Time spent on strategy optimization",
            ["optimizer_type"],
            registry=registry,
        )

        self.optimization_iterations = Counter(
            "tradepulse_optimization_iterations_total",
            "Total number of optimization iterations",
            ["optimizer_type"],
            registry=registry,
        )

    @property
    def enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def _resolve_status(self, ctx: Dict[str, Any], status: str) -> str:
        """Resolve the final status for metrics labels.

        When an exception occurs the contextual status must *always* be
        recorded as ``"error"`` regardless of any values stored on the
        context dictionary. For successful executions we still allow callers to
        provide a custom status value via ``ctx["status"]`` while gracefully
        handling ``None``/empty values by falling back to the default status.
        """

        if status == "error":
            return "error"

        override = ctx.get("status")
        if override is None:
            return status

        final_status = str(override).strip()
        return final_status or status

    @contextmanager
    def measure_feature_transform(
        self,
        feature_name: str,
        feature_type: str = "generic",
    ) -> Iterator[None]:
        """Context manager for measuring feature transformation time.

        Args:
            feature_name: Name of the feature being transformed
            feature_type: Type/category of the feature

        Example:
            >>> collector = MetricsCollector()
            >>> with collector.measure_feature_transform("RSI", "momentum"):
            ...     result = compute_rsi(prices)
        """
        if not self._enabled:
            yield
            return

        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.feature_transform_duration.labels(
                feature_name=feature_name, feature_type=feature_type
            ).observe(duration)
            self.feature_transform_total.labels(
                feature_name=feature_name, feature_type=feature_type, status=status
            ).inc()

    @contextmanager
    def measure_backtest(self, strategy: str) -> Iterator[Dict[str, Any]]:
        """Context manager for measuring backtest execution.

        Args:
            strategy: Name of the strategy being backtested

        Yields:
            Dictionary to store backtest results

        Example:
            >>> collector = MetricsCollector()
            >>> with collector.measure_backtest("momentum_strategy") as ctx:
            ...     result = run_backtest(...)
            ...     ctx["pnl"] = result.pnl
            ...     ctx["max_dd"] = result.max_dd
            ...     ctx["trades"] = result.trades
        """
        if not self._enabled:
            yield {}
            return

        start_time = time.time()
        status = "success"
        ctx: Dict[str, Any] = {}

        try:
            yield ctx
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.backtest_duration.labels(strategy=strategy).observe(duration)
            self.backtest_total.labels(strategy=strategy, status=status).inc()

            if status == "success" and ctx:
                if "pnl" in ctx:
                    self.backtest_pnl.labels(strategy=strategy).set(ctx["pnl"])
                if "max_dd" in ctx:
                    self.backtest_max_drawdown.labels(strategy=strategy).set(
                        abs(ctx["max_dd"])
                    )
                if "trades" in ctx:
                    self.backtest_trades.labels(strategy=strategy).set(ctx["trades"])

    def record_feature_value(self, feature_name: str, value: float) -> None:
        """Record a feature value.

        Args:
            feature_name: Name of the feature
            value: Feature value
        """
        if not self._enabled:
            return
        self.feature_value.labels(feature_name=feature_name).set(value)

    def _update_latency_quantiles(
        self,
        gauge: Gauge,
        labels: Dict[str, str],
        samples: deque[float],
    ) -> None:
        if not self._enabled or not samples:
            return
        values = list(map(float, samples))
        if not values:
            return

        quantiles = (0.5, 0.95, 0.99)
        if _NUMPY_AVAILABLE and np is not None and _accelerated_quantiles is not None:
            arr = np.fromiter(values, dtype=float, count=len(values))
            if arr.size == 0:
                return
            accelerated = _accelerated_quantiles(arr, quantiles)
            quantile_values = {
                q: float(value)
                for q, value in zip(quantiles, accelerated, strict=False)
            }
        else:
            quantile_values = _fallback_quantiles(values, quantiles)

        for quantile, name in zip(quantiles, ("p50", "p95", "p99")):
            value = quantile_values.get(quantile)
            if value is None:
                continue
            gauge.labels(**labels, quantile=name).set(value)

    @contextmanager
    def measure_signal_generation(self, strategy: str) -> Iterator[Dict[str, Any]]:
        """Measure latency of strategy signal generation."""

        if not self._enabled:
            yield {}
            return

        start_time = time.time()
        ctx: Dict[str, Any] = {}
        status = "success"

        try:
            yield ctx
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            final_status = self._resolve_status(ctx, status)
            samples = self._signal_latency_samples[strategy]
            samples.append(duration)
            self._update_latency_quantiles(
                self.signal_generation_latency_quantiles,
                {"strategy": strategy},
                samples,
            )
            self.signal_generation_total.labels(
                strategy=strategy, status=final_status
            ).inc()

    @contextmanager
    def measure_data_ingestion(
        self,
        source: str,
        symbol: str,
    ) -> Iterator[Dict[str, Any]]:
        """Context manager for measuring data ingestion operations.

        Args:
            source: Data source name
            symbol: Trading symbol

        Yields:
            Dictionary that can be populated with metadata (e.g. ``{"status": "error"}``).
        """
        if not self._enabled:
            yield {}
            return

        start_time = time.time()
        ctx: Dict[str, Any] = {}
        status = "success"

        try:
            yield ctx
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            final_status = self._resolve_status(ctx, status)
            self.data_ingestion_duration.labels(
                source=source,
                symbol=symbol,
            ).observe(duration)
            samples = self._ingestion_latency_samples[(source, symbol)]
            samples.append(duration)
            self._update_latency_quantiles(
                self.data_ingestion_latency_quantiles,
                {"source": source, "symbol": symbol},
                samples,
            )
            self.data_ingestion_total.labels(
                source=source,
                symbol=symbol,
                status=final_status,
            ).inc()

    def set_ingestion_throughput(
        self, source: str, symbol: str, throughput: float
    ) -> None:
        """Record instantaneous ingestion throughput."""

        if not self._enabled:
            return

        self.data_ingestion_throughput.labels(source=source, symbol=symbol).set(
            max(0.0, float(throughput))
        )

    def record_tick_processed(self, source: str, symbol: str, count: int = 1) -> None:
        """Record that ticks were processed.

        Args:
            source: Data source name
            symbol: Trading symbol
            count: Number of ticks processed in the batch
        """
        if not self._enabled:
            return
        self.ticks_processed.labels(source=source, symbol=symbol).inc(count)

    def record_order_placed(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        status: str = "success",
        count: int = 1,
    ) -> None:
        """Record an order placement.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            order_type: Order type (market, limit, etc.)
            status: Order status
            count: Number of orders placed
        """
        if not self._enabled:
            return
        self.orders_placed.labels(
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            status=status,
        ).inc(count)

    @contextmanager
    def measure_order_placement(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
    ) -> Iterator[Dict[str, Any]]:
        """Context manager for measuring order placement latency and outcomes."""

        if not self._enabled:
            yield {}
            return

        start_time = time.time()
        ctx: Dict[str, Any] = {}
        status = "success"

        try:
            yield ctx
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            final_status = self._resolve_status(ctx, status)
            self.order_placement_duration.labels(
                exchange=exchange,
                symbol=symbol,
            ).observe(duration)
            samples = self._order_submission_latency_samples[(exchange, symbol)]
            samples.append(duration)
            self._update_latency_quantiles(
                self.order_submission_latency_quantiles,
                {"exchange": exchange, "symbol": symbol},
                samples,
            )
            self.orders_placed.labels(
                exchange=exchange,
                symbol=symbol,
                order_type=order_type,
                status=final_status,
            ).inc()

    def set_open_positions(self, exchange: str, symbol: str, positions: float) -> None:
        """Update the gauge tracking open positions."""

        if not self._enabled:
            return
        self.open_positions.labels(exchange=exchange, symbol=symbol).set(positions)

    def record_order_fill_latency(
        self, exchange: str, symbol: str, duration: float
    ) -> None:
        """Observe latency from order submission to fill."""

        if not self._enabled:
            return
        samples = self._order_fill_latency_samples[(exchange, symbol)]
        samples.append(duration)
        self._update_latency_quantiles(
            self.order_fill_latency_quantiles,
            {"exchange": exchange, "symbol": symbol},
            samples,
        )

    def record_order_ack_latency(
        self, exchange: str, symbol: str, duration: float
    ) -> None:
        """Observe latency between order submission and venue acknowledgement."""

        if not self._enabled:
            return
        samples = self._order_ack_latency_samples[(exchange, symbol)]
        samples.append(duration)
        self._update_latency_quantiles(
            self.order_ack_latency_quantiles,
            {"exchange": exchange, "symbol": symbol},
            samples,
        )

    def record_signal_to_fill_latency(
        self,
        strategy: str,
        exchange: str,
        symbol: str,
        duration: float,
    ) -> None:
        """Observe latency from signal emission until the final fill completes."""

        if not self._enabled:
            return

        label_key = (strategy, exchange, symbol)
        samples = self._signal_to_fill_latency_samples[label_key]
        samples.append(duration)
        self._update_latency_quantiles(
            self.signal_to_fill_latency_quantiles,
            {"strategy": strategy, "exchange": exchange, "symbol": symbol},
            samples,
        )

    def set_strategy_score(self, strategy_name: str, score: float) -> None:
        """Record the latest strategy score."""

        if not self._enabled:
            return
        self.strategy_score.labels(strategy_name=strategy_name).set(score)

    def set_strategy_memory_size(self, size: int) -> None:
        """Update the number of strategies currently held in memory."""

        if not self._enabled:
            return
        self.strategy_memory_size.set(size)

    def record_regression_metrics(self, model: str, **metrics: float) -> None:
        """Record regression evaluation metrics for a given model identifier."""

        if not self._enabled:
            return
        for name, value in metrics.items():
            if value is None:
                continue
            self.regression_metrics.labels(model=model, metric=str(name)).set(
                float(value)
            )

    def record_equity_point(self, strategy: str, step: int, value: float) -> None:
        """Record a sample on the equity curve gauge."""

        if not self._enabled:
            return
        self.backtest_equity_curve.labels(strategy=strategy, step=str(step)).set(value)

    def record_environment_parity(
        self,
        *,
        strategy: str,
        status: str,
        deviations: Sequence["MetricDeviation"] | None = None,
    ) -> None:
        """Record the outcome of an environment parity evaluation."""

        if not self._enabled:
            return

        self.environment_parity_checks.labels(
            strategy=strategy, status=status
        ).inc()

        if not deviations:
            return

        for deviation in deviations:
            self.environment_parity_metric_deviation.labels(
                strategy=strategy,
                metric=deviation.metric,
                baseline=deviation.baseline_environment,
                comparison=deviation.comparison_environment,
            ).set(deviation.absolute_difference)

    def render_prometheus(self) -> str:
        """Render the currently collected metrics in Prometheus text format."""

        if not self._enabled:
            return ""

        payload = generate_latest(self.registry) if self.registry else generate_latest()
        return payload.decode("utf-8")

    def record_risk_validation(self, symbol: str, outcome: str) -> None:
        """Record the result of a risk validation."""

        if not self._enabled:
            return
        self.risk_validation_total.labels(symbol=symbol, outcome=outcome).inc()

    def record_kill_switch_trigger(self, reason: str) -> None:
        """Record a kill switch trigger occurrence."""

        if not self._enabled:
            return
        self.kill_switch_triggers_total.labels(reason=reason).inc()

    def record_compliance_check(
        self,
        symbol: str,
        status: str,
        violations: Iterable[str] | None = None,
    ) -> None:
        """Record the outcome of a compliance check."""

        if not self._enabled:
            return

        self.compliance_checks_total.labels(symbol=symbol, status=status).inc()
        if not violations:
            return
        for violation in violations:
            self.compliance_violations_total.labels(
                symbol=symbol,
                violation_type=str(violation),
            ).inc()

    def record_watchdog_restart(self, watchdog: str, worker: str) -> None:
        """Record a watchdog-driven worker restart."""

        if not self._enabled:
            return
        self.watchdog_worker_restarts.labels(watchdog=watchdog, worker=worker).inc()

    def set_watchdog_live_probe(self, watchdog: str, healthy: bool) -> None:
        """Update the outcome of the most recent watchdog live probe."""

        if not self._enabled:
            return
        self.watchdog_live_probe_status.labels(watchdog=watchdog).set(
            1.0 if healthy else 0.0
        )

    def set_watchdog_heartbeat(
        self, watchdog: str, timestamp: float | None = None
    ) -> None:
        """Record the timestamp associated with the latest watchdog heartbeat."""

        if not self._enabled:
            return
        if timestamp is None:
            timestamp = time.time()
        self.watchdog_last_heartbeat.labels(watchdog=watchdog).set(float(timestamp))

    def observe_health_check_latency(self, check_name: str, duration: float) -> None:
        """Observe the execution time of a health check probe."""

        if not self._enabled:
            return

        self.health_check_latency.labels(check_name=check_name).observe(
            max(0.0, float(duration))
        )

    def set_health_check_status(self, check_name: str, healthy: bool) -> None:
        """Update the status gauge tracking the latest health probe outcome."""

        if not self._enabled:
            return

        self.health_check_status.labels(check_name=check_name).set(
            1.0 if healthy else 0.0
        )


# Global metrics collector instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector(registry: Optional[Any] = None) -> MetricsCollector:
    """Get the global metrics collector instance.

    Args:
        registry: Prometheus registry (uses default if None)

    Returns:
        MetricsCollector instance
    """
    global _collector
    if _collector is None:
        _collector = MetricsCollector(registry)
    return _collector


def start_metrics_server(port: int = 8000, addr: str = "") -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to listen on
        addr: Address to bind to (empty string for all interfaces)
    """
    if not PROMETHEUS_AVAILABLE:
        raise RuntimeError("prometheus_client is not installed")
    start_http_server(port, addr)


def start_metrics_exporter_process(
    port: int = 8000, addr: str = ""
) -> multiprocessing.Process:
    """Spawn a Prometheus exporter in a dedicated process."""

    if not PROMETHEUS_AVAILABLE:
        raise RuntimeError("prometheus_client is not installed")

    from observability.exporters import start_prometheus_exporter_process

    return start_prometheus_exporter_process(port=port, addr=addr)


def stop_metrics_exporter_process(
    process: Optional[multiprocessing.Process], *, timeout: float = 5.0
) -> None:
    """Terminate a previously spawned Prometheus exporter process."""

    if process is None:
        return

    from observability.exporters import stop_exporter_process

    stop_exporter_process(process, timeout=timeout)


__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "start_metrics_server",
    "start_metrics_exporter_process",
    "stop_metrics_exporter_process",
]
