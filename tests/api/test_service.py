from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("TRADEPULSE_ADMIN_TOKEN", "import-admin-token")
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "import-audit-secret")

from application.api.service import create_app


@pytest.fixture()
def configured_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.delenv("TRADEPULSE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)
    return create_app(admin_token="unit-admin-token", audit_secret="unit-audit-secret")


def test_create_app_requires_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRADEPULSE_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="TRADEPULSE_ADMIN_TOKEN"):
        create_app()


def _build_payload() -> dict:
    base = datetime(2024, 12, 1, 12, 0, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for idx in range(60):
        ts = base + timedelta(minutes=idx)
        price += 0.1 if idx % 2 == 0 else -0.05
        bars.append(
            {
                "timestamp": ts.isoformat(),
                "open": price - 0.2,
                "high": price + 0.3,
                "low": price - 0.4,
                "close": price,
                "volume": 1000 + idx * 2,
                "bidVolume": 500 + idx,
                "askVolume": 480 + idx,
                "signedVolume": (-1) ** idx * 20.0,
            }
        )
    return {"symbol": "TEST-USD", "bars": bars}


def test_feature_endpoint_computes_latest_vector(configured_app: FastAPI) -> None:
    app = configured_app
    client = TestClient(app)

    payload = _build_payload()
    response = client.post("/features", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TEST-USD"
    assert "features" in body
    assert body["features"].get("macd") is not None
    assert response.headers["X-Cache-Status"] == "miss"
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert "Accept" in response.headers.get("Vary", "")

    # Second call should be served from the cache.
    cached_response = client.post("/features", json=payload)
    assert cached_response.headers["X-Cache-Status"] == "hit"
    assert cached_response.json() == body


def test_prediction_endpoint_returns_signal(configured_app: FastAPI) -> None:
    app = configured_app
    client = TestClient(app)

    payload = _build_payload()
    payload["horizon_seconds"] = 900

    response = client.post("/predictions", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TEST-USD"
    assert body["horizon_seconds"] == 900
    signal = body["signal"]
    assert signal["symbol"] == "TEST-USD"
    assert 0.0 <= signal["confidence"] <= 1.0
    assert "score" in signal["metadata"]
    assert response.headers["X-Cache-Status"] == "miss"
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert "Accept" in response.headers.get("Vary", "")

    cached = client.post("/predictions", json=payload)
    assert cached.headers["X-Cache-Status"] == "hit"
    assert cached.json() == body


def test_admin_endpoints_set_strict_cache_headers(configured_app: FastAPI) -> None:
    app = configured_app
    client = TestClient(app)
    headers = {"X-Admin-Token": "unit-admin-token"}

    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"
    vary_header = response.headers.get("Vary", "")
    assert "X-Admin-Token" in vary_header
    assert "Accept" in vary_header
