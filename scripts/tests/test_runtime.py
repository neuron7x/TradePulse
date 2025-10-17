from __future__ import annotations

# SPDX-License-Identifier: MIT
import io
import os
import threading
import time
from pathlib import Path

import pytest

from scripts.runtime import (
    ChecksumMismatchError,
    ProgressBar,
    compute_checksum,
    create_artifact_manager,
    transfer_with_resume,
    verify_checksum,
)
from scripts.runtime.pathfinder import find_resources
from scripts.runtime.task_queue import task_queue


def test_artifact_manager_creates_timestamped_directory(tmp_path: Path) -> None:
    manager = create_artifact_manager("demo", root=tmp_path)
    path = manager.directory
    assert path.parent == tmp_path / "demo"
    # Timestamp formatted like 20240101T000000Z
    assert path.name.endswith("Z")


def test_checksum_roundtrip(tmp_path: Path) -> None:
    sample = tmp_path / "payload.bin"
    sample.write_bytes(b"tradepulse" * 8)
    digest = compute_checksum(sample)
    verify_checksum(sample, digest)
    with pytest.raises(ChecksumMismatchError):
        verify_checksum(sample, "deadbeef")


def test_transfer_with_resume_local_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    payload = os.urandom(128 * 1024)
    source.write_bytes(payload)

    destination = tmp_path / "dest.bin"
    # Simulate an interrupted transfer by copying the first half
    destination.write_bytes(payload[: len(payload) // 2])

    progress_stream = io.StringIO()
    progress = ProgressBar(total=None, label="transfer", stream=progress_stream)
    transfer_with_resume(source, destination, progress=progress)

    assert destination.read_bytes() == payload


def test_task_queue_limits_parallelism(tmp_path: Path) -> None:
    # Ensure that the task_queue helper executes tasks respecting the worker limit.
    lock = threading.Lock()
    active = 0
    peak = 0

    def _worker(duration: float) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(duration)
        with lock:
            active -= 1

    with task_queue(max_workers=2) as queue:
        queue.submit(_worker, 0.1)
        queue.submit(_worker, 0.1)
        queue.submit(_worker, 0.1)

    assert peak <= 2


def test_find_resources(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.txt").write_text("hello", encoding="utf-8")
    (nested / "b.txt").write_text("world", encoding="utf-8")

    results = list(find_resources("*.txt", [nested]))
    assert {path.name for path in results} == {"a.txt", "b.txt"}
