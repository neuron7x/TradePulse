# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from core.data.streaming import RollingBuffer


def test_rolling_buffer_retains_last_elements() -> None:
    buf = RollingBuffer(size=3)
    buf.extend([1.0, 2.0, 3.0, 4.0])
    assert buf.values() == [2.0, 3.0, 4.0]
    assert buf.is_full()
    assert len(buf.values()) == 3


def test_rolling_buffer_handles_smaller_sequences() -> None:
    buf = RollingBuffer(size=5)
    buf.push(1.0)
    buf.push(2)
    assert buf.values() == [1.0, 2.0]
    assert buf.last() == 2.0
    assert not buf.is_full()


def test_rolling_buffer_streaming_statistics() -> None:
    buf = RollingBuffer(size=4)
    buf.extend([1, 2, 3])
    assert buf.mean() == 2.0
    assert pytest.approx(buf.std(), rel=1e-12) == math.sqrt(2 / 3)
    assert pytest.approx(buf.std(ddof=1), rel=1e-12) == 1.0
    buf.push(5)
    assert buf.values() == [1.0, 2.0, 3.0, 5.0][-4:]
    assert pytest.approx(buf.mean(), rel=1e-12) == 2.75
    assert pytest.approx(buf.std(), rel=1e-12) == math.sqrt(2.1875)


def test_clear_resets_state() -> None:
    buf = RollingBuffer(size=2)
    buf.extend([10.0, 20.0])
    buf.clear()
    assert buf.values() == []
    buf.push(5)
    assert buf.values() == [5.0]
