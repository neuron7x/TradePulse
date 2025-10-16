from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rate_limit_enforces_quota(async_api_client, admin_token_factory, feature_payload):
    token = admin_token_factory(subject="rate-limited-user")
    headers = {"Authorization": f"Bearer {token}"}

    responses = []
    for _ in range(3):
        responses.append(await async_api_client.post("/features", json=feature_payload, headers=headers))
        await asyncio.sleep(0.01)

    status_codes = [response.status_code for response in responses]
    assert status_codes.count(429) >= 1
    throttled = next(resp for resp in responses if resp.status_code == 429)
    payload = throttled.json()
    assert payload["error"]["code"] == 429
    assert "Rate limit" in payload["error"]["message"]
