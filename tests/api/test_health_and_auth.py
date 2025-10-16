from __future__ import annotations

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_endpoint_reports_ready(async_api_client):
    response = await async_api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "components" in body
    assert set(body["components"]).issuperset({"risk_manager", "inference_cache", "client_rate_limiter"})


@pytest.mark.asyncio
@pytest.mark.integration
async def test_health_endpoint_reflects_cache_metrics(async_api_client, admin_token_factory, feature_payload):
    token = admin_token_factory(subject="health-cache-user")
    headers = {"Authorization": f"Bearer {token}"}

    # Warm the cache to ensure utilisation metrics are populated.
    for _ in range(2):
        res = await async_api_client.post("/features", json=feature_payload, headers=headers)
        assert res.status_code == 200

    response = await async_api_client.get("/health")
    assert response.status_code in (200, 503)
    cache_metrics = response.json()["components"]["inference_cache"]["metrics"]
    assert cache_metrics["entries"] >= 1
    assert cache_metrics["max_entries"] >= cache_metrics["entries"]
