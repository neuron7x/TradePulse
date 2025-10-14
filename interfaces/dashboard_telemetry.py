# SPDX-License-Identifier: MIT
"""Telemetry helpers for the dashboard UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

from observability.tracing import current_traceparent


def post_metrics(
    url: str,
    payload: dict[str, Any],
    *,
    post: Callable[..., requests.Response] | None = None,
) -> tuple[bool, str]:
    """Send metrics to an API endpoint, propagating the active traceparent."""

    headers = {"Content-Type": "application/json"}
    enriched_payload = dict(payload)
    traceparent = current_traceparent()
    if traceparent:
        headers["traceparent"] = traceparent
        enriched_payload.setdefault("traceparent", traceparent)

    post_call = post or requests.post

    try:
        response = post_call(url, json=enriched_payload, headers=headers, timeout=5)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        return False, f"Failed to POST metrics: {exc}"

    if 200 <= response.status_code < 300:
        return True, "Metrics successfully sent to backend."
    return False, f"Backend responded with status {response.status_code}: {response.text[:200]}"
