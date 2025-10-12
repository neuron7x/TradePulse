import logging

import pytest

import observability.telemetry as telemetry


@pytest.mark.skipif(not telemetry._OTEL_AVAILABLE, reason="OpenTelemetry optional extras not installed")
def test_configure_telemetry_sets_metrics_and_logs(monkeypatch):
    monkeypatch.setattr(telemetry, "configure_tracing", lambda cfg=None: True)
    monkeypatch.setattr(telemetry, "_METRIC_PROVIDER", None, raising=False)
    monkeypatch.setattr(telemetry, "_LOGGER_PROVIDER", None, raising=False)
    monkeypatch.setattr(telemetry, "_LOGGING_HANDLER", None, raising=False)

    class DummyMetricExporter:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    class DummyLogExporter:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    class DummyReader:  # pragma: no cover - simple stub
        def __init__(self, exporter, export_interval_millis: int) -> None:
            self.exporter = exporter
            self.interval = export_interval_millis

    class DummyMeterProvider:
        def __init__(self, *, resource=None, metric_readers=None) -> None:
            self.resource = resource
            self.metric_readers = list(metric_readers or [])

    class DummyProcessor:  # pragma: no cover - simple stub
        def __init__(self, exporter) -> None:
            self.exporter = exporter

    class DummyLoggerProvider:
        def __init__(self, resource=None) -> None:
            self.resource = resource
            self.processors: list[DummyProcessor] = []

        def add_log_record_processor(self, processor) -> None:
            self.processors.append(processor)

    class DummyLoggingHandler(logging.Handler):
        def __init__(self, *, level=logging.NOTSET, logger_provider=None) -> None:
            super().__init__(level)
            self.logger_provider = logger_provider

    captured_provider: dict[str, object] = {}

    def dummy_set_logger_provider(provider) -> None:
        captured_provider["provider"] = provider

    monkeypatch.setattr(telemetry, "OTLPMetricExporter", DummyMetricExporter)
    monkeypatch.setattr(telemetry, "OTLPLogExporter", DummyLogExporter)
    monkeypatch.setattr(telemetry, "PeriodicExportingMetricReader", DummyReader)
    monkeypatch.setattr(telemetry, "MeterProvider", DummyMeterProvider)
    monkeypatch.setattr(telemetry, "BatchLogRecordProcessor", DummyProcessor)
    monkeypatch.setattr(telemetry, "LoggerProvider", DummyLoggerProvider)
    monkeypatch.setattr(telemetry, "LoggingHandler", DummyLoggingHandler)
    monkeypatch.setattr(telemetry, "set_logger_provider", dummy_set_logger_provider)

    root_logger = logging.getLogger()
    existing_handlers = set(root_logger.handlers)

    status = telemetry.configure_telemetry(
        telemetry.TelemetryConfig(
            service_name="tradepulse-test",
            environment="ci",
            metrics=telemetry.MetricsConfig(collection_interval=5.0),
            logging=telemetry.LoggingConfig(),
        )
    )

    assert status.tracing is True
    assert status.metrics is True
    assert status.logging is True
    assert isinstance(telemetry._METRIC_PROVIDER, DummyMeterProvider)
    assert telemetry._METRIC_PROVIDER.metric_readers[0].interval == 5000
    assert isinstance(captured_provider.get("provider"), DummyLoggerProvider)

    new_handlers = [handler for handler in root_logger.handlers if handler not in existing_handlers]
    try:
        assert any(isinstance(handler, DummyLoggingHandler) for handler in new_handlers)
    finally:
        for handler in new_handlers:
            root_logger.removeHandler(handler)


def test_configure_telemetry_graceful_when_otel_missing(monkeypatch):
    monkeypatch.setattr(telemetry, "configure_tracing", lambda cfg=None: False)
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", False, raising=False)
    monkeypatch.setattr(telemetry, "otel_metrics", None, raising=False)
    monkeypatch.setattr(telemetry, "MeterProvider", None, raising=False)
    monkeypatch.setattr(telemetry, "PeriodicExportingMetricReader", None, raising=False)
    monkeypatch.setattr(telemetry, "OTLPMetricExporter", None, raising=False)
    monkeypatch.setattr(telemetry, "LoggerProvider", None, raising=False)
    monkeypatch.setattr(telemetry, "LoggingHandler", None, raising=False)
    monkeypatch.setattr(telemetry, "BatchLogRecordProcessor", None, raising=False)
    monkeypatch.setattr(telemetry, "OTLPLogExporter", None, raising=False)
    monkeypatch.setattr(telemetry, "set_logger_provider", lambda *_: None, raising=False)

    status = telemetry.configure_telemetry(telemetry.TelemetryConfig())
    assert status.tracing is False
    assert status.metrics is False
    assert status.logging is False
