from __future__ import annotations

import httpx
import pytest

from core.data.adapters.polygon import PolygonIngestionAdapter


@pytest.mark.asyncio
@pytest.mark.integration
async def test_polygon_adapter_fetch_parses_ticks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v2/aggs/ticker/AAPL/range/1/minute/2024-01-01/2024-01-02"
        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"t": 1704067200000, "c": 123.45, "v": 1500},
                    {"t": 1704067260000, "c": 123.55, "v": 1600},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://api.polygon.io", transport=transport) as client:
        adapter = PolygonIngestionAdapter(api_key="test-key", client=client)
        ticks = await adapter.fetch(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-02",
            multiplier=1,
            timespan="minute",
        )
        await adapter.aclose()

    assert len(ticks) == 2
    first = ticks[0]
    assert first.symbol == "AAPL"
    assert float(first.price) == pytest.approx(123.45)
    assert float(first.volume) == pytest.approx(1500)
