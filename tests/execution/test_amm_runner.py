from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from execution.amm_runner import AMMRunner


def test_on_tick_publishes_metrics(monkeypatch: pytest.MonkeyPatch):
    started_ports = []

    def fake_start_http_server(port: int) -> None:
        started_ports.append(port)

    published = {}

    def fake_publish(symbol: str, tf: str, out: dict, k: float, theta: float, q_hi):
        published.setdefault("calls", []).append((symbol, tf, out, k, theta, q_hi))

    @contextmanager
    def fake_timed(symbol: str, tf: str):
        published.setdefault("timed", []).append((symbol, tf))
        yield

    monkeypatch.setattr("execution.amm_runner.start_http_server", fake_start_http_server)
    monkeypatch.setattr("execution.amm_runner.publish_metrics", fake_publish)
    monkeypatch.setattr("execution.amm_runner.timed_update", fake_timed)

    runner = AMMRunner("BTC", "1m")

    async def go() -> dict:
        return await runner.on_tick(0.01, 0.6, 0.1, None)

    result = asyncio.run(go())
    assert started_ports == [9095]
    assert set(result) == {"amm_pulse", "amm_precision", "amm_valence", "pred", "pe", "entropy"}
    assert published["timed"] == [("BTC", "1m")]
    assert published["calls"][0][0:2] == ("BTC", "1m")


def test_run_stream_yields_results(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("execution.amm_runner.start_http_server", lambda port: None)
    monkeypatch.setattr("execution.amm_runner.publish_metrics", lambda *args, **kwargs: None)

    @contextmanager
    def fake_timed(symbol: str, tf: str):
        yield

    monkeypatch.setattr("execution.amm_runner.timed_update", fake_timed)

    runner = AMMRunner("ETH", "5m")

    async def ticker():
        for i in range(3):
            yield (0.005 * i, 0.5, (-1) ** i * 0.1, None)

    async def collect():
        results = []
        async for out in runner.run_stream(ticker()):
            results.append(out)
        return results

    outputs = asyncio.run(collect())
    assert len(outputs) == 3
    for item in outputs:
        assert isinstance(item["amm_pulse"], float)
