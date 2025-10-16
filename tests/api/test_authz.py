from __future__ import annotations

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_features_endpoint_requires_bearer(async_api_client, feature_payload):
    response = await async_api_client.post("/features", json=feature_payload)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == 401
    assert body["error"]["message"] == "Bearer token required for this endpoint."


@pytest.mark.asyncio
@pytest.mark.integration
async def test_features_endpoint_accepts_valid_token(async_api_client, admin_token_factory, feature_payload):
    token = admin_token_factory(subject="feature-user")
    headers = {"Authorization": f"Bearer {token}"}

    first = await async_api_client.post("/features", json=feature_payload, headers=headers)
    assert first.status_code == 200
    assert first.headers["X-Cache-Status"] == "miss"
    body = first.json()
    assert body["symbol"] == feature_payload["symbol"]
    assert set(body["features"]).issuperset({"macd", "macd_signal", "macd_histogram"})

    cached = await async_api_client.post("/features", json=feature_payload, headers=headers)
    assert cached.status_code == 200
    assert cached.headers["X-Cache-Status"] == "hit"
    assert cached.json() == body
