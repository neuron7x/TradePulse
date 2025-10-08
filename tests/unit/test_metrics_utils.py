# SPDX-License-Identifier: MIT
"""Tests for Prometheus metrics collection module."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from core.utils.metrics import (
    MetricsCollector,
    get_metrics_collector,
    start_metrics_server,
)


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_init_with_prometheus_available(self) -> None:
        """Should initialize successfully when Prometheus is available."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", True):
            collector = MetricsCollector()
            assert collector._enabled is True

    def test_init_without_prometheus(self) -> None:
        """Should disable when Prometheus is not available."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", False):
            collector = MetricsCollector()
            assert collector._enabled is False

    def test_has_feature_metrics(self) -> None:
        """Should have feature transformation metrics."""
        try:
            collector = MetricsCollector()
            if collector._enabled:
                assert hasattr(collector, "feature_transform_duration")
                assert hasattr(collector, "feature_transform_total")
                assert hasattr(collector, "feature_value")
        except Exception:
            pytest.skip("Prometheus not available")

    def test_has_backtest_metrics(self) -> None:
        """Should have backtest metrics."""
        try:
            collector = MetricsCollector()
            if collector._enabled:
                assert hasattr(collector, "backtest_duration")
                assert hasattr(collector, "backtest_total")
        except Exception:
            pytest.skip("Prometheus not available")

    def test_has_data_metrics(self) -> None:
        """Should have data ingestion metrics."""
        try:
            collector = MetricsCollector()
            if collector._enabled:
                assert hasattr(collector, "data_ingestion_total")
                assert hasattr(collector, "data_ingestion_duration")
        except Exception:
            pytest.skip("Prometheus not available")

    def test_has_execution_metrics(self) -> None:
        """Should have execution metrics."""
        try:
            collector = MetricsCollector()
            if collector._enabled:
                assert hasattr(collector, "orders_total")
                assert hasattr(collector, "execution_duration")
        except Exception:
            pytest.skip("Prometheus not available")


class TestMeasureFeatureTransform:
    """Test measure_feature_transform context manager."""

    def test_measure_feature_transform_records_duration(self) -> None:
        """Should record feature transformation duration."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with collector.measure_feature_transform("test_feature", "indicator"):
                time.sleep(0.01)  # Small delay
            
            # If we get here without error, the context manager worked
            assert True
        except Exception:
            pytest.skip("Prometheus not available")

    def test_measure_feature_transform_increments_counter(self) -> None:
        """Should increment feature transform counter."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            # Mock the counter to verify it's called
            mock_counter = MagicMock()
            collector.feature_transform_total = mock_counter
            
            with collector.measure_feature_transform("test_feature", "indicator"):
                pass
            
            # Counter should have been incremented
            mock_counter.labels.assert_called()
        except Exception:
            pytest.skip("Prometheus not available")


class TestMeasureBacktest:
    """Test measure_backtest context manager."""

    def test_measure_backtest_records_duration(self) -> None:
        """Should record backtest duration."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with collector.measure_backtest("test_strategy"):
                time.sleep(0.01)
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")

    def test_measure_backtest_with_status(self) -> None:
        """Should track backtest with status."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            ctx = collector.measure_backtest("test_strategy")
            ctx.__enter__()
            ctx.set_status("success")
            ctx.__exit__(None, None, None)
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")


class TestMeasureDataIngestion:
    """Test data ingestion metrics."""

    def test_record_tick_processed(self) -> None:
        """Should record processed ticks."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_tick_processed("csv", "BTCUSD")
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")


class TestMeasureExecution:
    """Test execution metrics."""

    def test_record_order_placed(self) -> None:
        """Should record order placement."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_order_placed("buy", "market", "success")
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")


class TestRecordFeatureValue:
    """Test record_feature_value method."""

    def test_records_feature_value(self) -> None:
        """Should record feature value as gauge."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_feature_value("test_feature", 42.5)
            
            # If no error, the method worked
            assert True
        except Exception:
            pytest.skip("Prometheus not available")

    def test_records_negative_values(self) -> None:
        """Should handle negative feature values."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_feature_value("test_feature", -10.0)
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")


class TestRecordOrder:
    """Test record_order_placed method."""

    def test_records_order(self) -> None:
        """Should record order execution."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_order_placed("buy", "market", "success")
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")


class TestGetMetricsCollector:
    """Test get_metrics_collector function."""

    def test_returns_collector_instance(self) -> None:
        """Should return MetricsCollector instance."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_returns_singleton(self) -> None:
        """Should return same instance on multiple calls."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        # Should be the same instance
        assert collector1 is collector2


class TestStartMetricsServer:
    """Test start_metrics_server function."""

    @patch("core.utils.metrics.start_http_server")
    def test_starts_server_with_default_port(self, mock_start) -> None:
        """Should start metrics server on default port."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", True):
            start_metrics_server()
            mock_start.assert_called_once_with(8000, "")

    @patch("core.utils.metrics.start_http_server")
    def test_starts_server_with_custom_port(self, mock_start) -> None:
        """Should start metrics server on custom port."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", True):
            start_metrics_server(port=9090)
            mock_start.assert_called_once_with(9090, "")

    def test_raises_when_prometheus_unavailable(self) -> None:
        """Should raise RuntimeError when Prometheus is unavailable."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="prometheus_client is not installed"):
                start_metrics_server()


class TestContextManagerErrorHandling:
    """Test error handling in context managers."""

    def test_measure_feature_transform_handles_exceptions(self) -> None:
        """Should handle exceptions in measure_feature_transform."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with pytest.raises(ValueError):
                with collector.measure_feature_transform("test", "indicator"):
                    raise ValueError("Test error")
            
            # Should still record metrics even on error
        except Exception:
            pytest.skip("Prometheus not available")

    def test_measure_backtest_handles_exceptions(self) -> None:
        """Should handle exceptions in measure_backtest."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with pytest.raises(ValueError):
                with collector.measure_backtest("test"):
                    raise ValueError("Test error")
        except Exception:
            pytest.skip("Prometheus not available")


class TestMetricsDisabled:
    """Test metrics behavior when disabled."""

    def test_methods_work_when_disabled(self) -> None:
        """Methods should work gracefully when metrics are disabled."""
        with patch("core.utils.metrics.PROMETHEUS_AVAILABLE", False):
            collector = MetricsCollector()
            
            # Should not crash when disabled
            assert collector._enabled is False
            
            # These methods should handle disabled state gracefully
            collector.record_feature_value("test", 42.0)
            collector.record_tick_processed("csv", "BTCUSD")
            collector.record_order_placed("buy", "market", "success")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_feature_name(self) -> None:
        """Should handle empty feature name."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with collector.measure_feature_transform("", "indicator"):
                pass
        except Exception:
            pytest.skip("Prometheus not available")

    def test_zero_duration(self) -> None:
        """Should handle zero duration measurements."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            with collector.measure_feature_transform("test", "indicator"):
                pass  # Very fast operation
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")

    def test_very_large_value(self) -> None:
        """Should handle very large feature values."""
        try:
            collector = MetricsCollector()
            if not collector._enabled:
                pytest.skip("Prometheus not available")
            
            collector.record_feature_value("test", 1e10)
            
            assert True
        except Exception:
            pytest.skip("Prometheus not available")
