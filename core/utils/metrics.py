# SPDX-License-Identifier: MIT
"""Prometheus metrics collection for TradePulse.

This module provides instrumentation for all critical entrypoints and
performance-sensitive operations.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        start_http_server,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


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
        
        self.ticks_processed = Counter(
            "tradepulse_ticks_processed_total",
            "Total number of ticks processed",
            ["source", "symbol"],
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

        # Generic operation metrics
        self.operation_duration = Histogram(
            "tradepulse_operation_duration_seconds",
            "Time spent in observability scopes",
            ["component", "operation"],
            registry=registry,
        )

        self.operation_total = Counter(
            "tradepulse_operation_total",
            "Total number of observability-scoped operations",
            ["component", "operation", "status"],
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
                feature_name=feature_name,
                feature_type=feature_type
            ).observe(duration)
            self.feature_transform_total.labels(
                feature_name=feature_name,
                feature_type=feature_type,
                status=status
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
                    self.backtest_max_drawdown.labels(strategy=strategy).set(abs(ctx["max_dd"]))
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
            self.data_ingestion_total.labels(
                source=source,
                symbol=symbol,
                status=final_status,
            ).inc()

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

    @contextmanager
    def measure_operation(
        self, component: str, operation: str
    ) -> Iterator[Dict[str, Any]]:
        """Track observability operation timing and status."""

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
            final_status = self._resolve_status(ctx, status)
            self.operation_duration.labels(
                component=component,
                operation=operation,
            ).observe(duration)
            self.operation_total.labels(
                component=component,
                operation=operation,
                status=final_status,
            ).inc()

    def render_prometheus(self) -> str:
        """Render the currently collected metrics in Prometheus text format."""

        if not self._enabled:
            return ""

        payload = generate_latest(self.registry) if self.registry else generate_latest()
        return payload.decode("utf-8")


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


__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "start_metrics_server",
]
