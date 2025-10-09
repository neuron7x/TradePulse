# SPDX-License-Identifier: MIT
"""Unified observability configuration for TradePulse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.utils.logging import configure_logging, get_logger
from core.utils.metrics import get_metrics_collector, start_metrics_server

from .context import ObservabilityScope, observability_scope
from .tracing import TracingConfig, configure_tracing, is_tracing_enabled


@dataclass
class ObservabilityConfig:
    """Configuration options for unified observability bootstrapping."""

    logging_level: str = "INFO"
    logging_json: bool = True
    metrics_enabled: bool = True
    metrics_port: Optional[int] = 8000
    metrics_addr: str = ""
    metrics_auto_start: bool = True
    tracing: Optional[TracingConfig] = None
    service_name: str = "tradepulse"


def bootstrap_observability(config: ObservabilityConfig) -> None:
    """Initialize logging, metrics, and tracing according to ``config``."""

    configure_logging(level=config.logging_level, use_json=config.logging_json)
    bootstrap_logger = get_logger(__name__)

    if config.metrics_enabled:
        collector = get_metrics_collector()
        bootstrap_logger.info(
            "Metrics collector initialized",
            metrics_enabled=collector.enabled,
            port=config.metrics_port,
            addr=config.metrics_addr,
        )

        if config.metrics_auto_start and config.metrics_port is not None:
            try:
                start_metrics_server(port=config.metrics_port, addr=config.metrics_addr)
                bootstrap_logger.info(
                    "Prometheus metrics server started",
                    port=config.metrics_port,
                    addr=config.metrics_addr,
                )
            except RuntimeError as exc:
                bootstrap_logger.warning(
                    "Prometheus metrics requested but unavailable",
                    error=str(exc),
                )

    if config.tracing is not None:
        tracing_config = config.tracing
        if tracing_config.service_name == "tradepulse" and config.service_name:
            tracing_config = TracingConfig(
                service_name=config.service_name,
                exporter=tracing_config.exporter,
                endpoint=tracing_config.endpoint,
                headers=tracing_config.headers,
                insecure=tracing_config.insecure,
                sample_ratio=tracing_config.sample_ratio,
                console_debug=tracing_config.console_debug,
            )

        try:
            configure_tracing(tracing_config)
            bootstrap_logger.info(
                "Tracing enabled",
                exporter=tracing_config.exporter,
                endpoint=tracing_config.endpoint,
            )
        except RuntimeError as exc:
            bootstrap_logger.warning(
                "Tracing configuration skipped",
                error=str(exc),
            )


__all__ = [
    "ObservabilityConfig",
    "ObservabilityScope",
    "bootstrap_observability",
    "observability_scope",
    "TracingConfig",
    "configure_tracing",
    "is_tracing_enabled",
]

