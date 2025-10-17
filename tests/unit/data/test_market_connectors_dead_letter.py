# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

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


def test_dead_letter_queue_persist_requires_destination(tmp_path: Path) -> None:
    queue = DeadLetterQueue(max_items=2)
    queue.push({"id": 1}, "error", context="persist")

    with pytest.raises(ValueError):
        queue.persist()

    target = tmp_path / "dead_letters.json"
    queue.persist(target)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["context"] == "persist"
    # Persist without drain keeps items in memory for inspection
    assert queue.peek()


def test_dead_letter_queue_persist_with_drain(tmp_path: Path) -> None:
    queue = DeadLetterQueue(max_items=2)
    queue.push({"id": 1}, "error", context="drain")
    queue.push({"id": 2}, "error", context="drain")

    target = tmp_path / "dead_letters_drain.json"
    queue.persist(target, drain=True)
    assert queue.peek() == []
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert [item["payload"] for item in saved] == [{"id": 1}, {"id": 2}]


def test_dead_letter_queue_appends_to_persistent_path(tmp_path: Path) -> None:
    path = tmp_path / "dead_letters.ndjson"
    queue = DeadLetterQueue(max_items=3, persistent_path=path)
    queue.push({"id": 1}, "boom", context="fetch")
    queue.push({"id": 2}, "boom", context="stream")

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["context"] == "fetch"
    assert records[1]["context"] == "stream"

    queue.persist(drain=True)
    assert queue.peek() == []
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert len(snapshot) == 2
    assert snapshot[0]["context"] == "fetch"
