# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from core.data.connectors.market import DeadLetterItem, DeadLetterQueue


class _ModelDumping:
    def __init__(self, value: int) -> None:
        self.value = value

    def model_dump(self) -> dict[str, int]:
        return {"value": self.value}


class _CustomObject:
    def __repr__(self) -> str:
        return "<custom>"


def test_dead_letter_queue_rejects_non_positive_max_items() -> None:
    with pytest.raises(ValueError):
        DeadLetterQueue(max_items=0)


def test_dead_letter_queue_push_and_peek_with_model_dump() -> None:
    queue = DeadLetterQueue(max_items=2)
    queue.push(_ModelDumping(5), ValueError("boom"), context="snapshot")
    peeked = queue.peek()
    assert len(peeked) == 1
    assert isinstance(peeked[0], DeadLetterItem)
    assert peeked[0].payload == {"value": 5}


def test_dead_letter_queue_push_with_custom_object() -> None:
    queue = DeadLetterQueue(max_items=2)
    queue.push(_CustomObject(), "error", context="stream")
    assert queue.peek()[0].payload == "<custom>"


def test_dead_letter_queue_drain_clears_items() -> None:
    queue = DeadLetterQueue(max_items=2)
    queue.push({"id": 1}, "error", context="test")
    drained = queue.drain()
    assert drained
    assert queue.peek() == []
