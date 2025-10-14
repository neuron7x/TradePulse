"""Unit tests for dashboard telemetry helpers."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from interfaces import dashboard_telemetry as telemetry


class _Response:
    def __init__(self, status_code: int, text: str = "", **extra: Any) -> None:
        self.status_code = status_code
        self.text = text
        for name, value in extra.items():
            setattr(self, name, value)


def test_post_metrics_enriches_traceparent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response(204)

    monkeypatch.setattr(telemetry, "current_traceparent", lambda: "00-abc-xyz")

    success, message = telemetry.post_metrics(
        "https://backend.example/api",
        {"R": 0.92},
        post=fake_post,
    )

    assert success
    assert "successfully" in message
    assert captured["url"] == "https://backend.example/api"
    assert captured["json"]["R"] == 0.92
    assert captured["headers"]["traceparent"] == "00-abc-xyz"
    assert captured["json"]["traceparent"] == "00-abc-xyz"


def test_post_metrics_handles_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_post(*_: Any, **__: Any) -> _Response:
        raise requests.Timeout("boom")

    monkeypatch.setattr(telemetry, "current_traceparent", lambda: None)

    success, message = telemetry.post_metrics(
        "https://backend.example/api",
        {"H": 0.12},
        post=failing_post,
    )

    assert not success
    assert "Failed to POST metrics" in message
