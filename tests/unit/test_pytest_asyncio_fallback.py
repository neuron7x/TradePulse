# SPDX-License-Identifier: MIT
from __future__ import annotations

import asyncio
import importlib

import pytest

conftest = importlib.import_module("conftest")


HAS_PYTEST_ASYNCIO = getattr(conftest, "HAS_PYTEST_ASYNCIO", False)
pytestmark = pytest.mark.skipif(
    HAS_PYTEST_ASYNCIO,
    reason="pytest-asyncio plugin available; fallback not exercised",
)

run_coroutine = importlib.import_module("tools.pytest_asyncio_fallback").run_coroutine


async def _loop_echo(loop: asyncio.AbstractEventLoop) -> None:
    assert asyncio.get_running_loop() is loop


def test_run_coroutine_uses_event_loop_fixture() -> None:
    loop = asyncio.new_event_loop()
    try:
        previous_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(previous_loop)
            run_coroutine(_loop_echo(loop), loop=loop)
            assert asyncio.get_event_loop() is previous_loop
        finally:
            previous_loop.run_until_complete(previous_loop.shutdown_asyncgens())
            previous_loop.close()
            asyncio.set_event_loop(None)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def _toggle_flag(flag: dict[str, bool]) -> None:
    flag["ran"] = True


def test_pyfunc_call_without_event_loop_fixture() -> None:
    flag = {"ran": False}
    
    async def _runner() -> None:
        await _toggle_flag(flag)

    run_coroutine(_runner())
    assert flag["ran"] is True
