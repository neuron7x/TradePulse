# SPDX-License-Identifier: MIT
"""Helpers that emulate a subset of ``pytest-asyncio``'s behaviour."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_coroutine(coroutine: Coroutine[Any, Any, object], *, loop=None) -> None:
    """Execute ``coroutine`` using ``loop`` when supplied."""

    if loop is None:
        asyncio.run(coroutine)
        return

    if not isinstance(loop, asyncio.AbstractEventLoop):
        msg = "event_loop fixture must provide an asyncio event loop instance"
        raise TypeError(msg)

    if loop.is_closed():
        raise RuntimeError("event_loop fixture provided a closed loop")

    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None

    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coroutine)
    finally:
        asyncio.set_event_loop(previous_loop)
