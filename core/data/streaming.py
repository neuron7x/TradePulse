# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections import deque
import math
from typing import Deque, Iterable, Iterator


class RollingBuffer:
    """Fixed-width numerical buffer with streaming statistics.

    The implementation keeps a running sum and sum of squares so aggregate
    metrics such as the mean or standard deviation can be queried in constant
    time without copying the underlying data. The buffer accepts any numerical
    input that can be coerced to ``float``. Values are stored in insertion
    order and the oldest item is discarded once ``size`` elements have been
    pushed.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be a positive integer")
        self._maxlen = int(size)
        self._buffer: Deque[float] = deque(maxlen=self._maxlen)
        self._sum = 0.0
        self._sum_sq = 0.0

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._buffer)

    def __iter__(self) -> Iterator[float]:
        return iter(self._buffer)

    def clear(self) -> None:
        """Remove all elements and reset the cached aggregates."""

        self._buffer.clear()
        self._sum = 0.0
        self._sum_sq = 0.0

    def is_full(self) -> bool:
        return len(self._buffer) == self._maxlen

    def push(self, value: float) -> None:
        value = float(value)
        if self.is_full():
            oldest = self._buffer[0]
            self._sum -= oldest
            self._sum_sq -= oldest * oldest
        self._buffer.append(value)
        self._sum += value
        self._sum_sq += value * value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.push(value)

    def values(self) -> list[float]:
        return list(self._buffer)

    def last(self) -> float:
        if not self._buffer:
            raise IndexError("RollingBuffer is empty")
        return self._buffer[-1]

    def mean(self) -> float:
        if not self._buffer:
            raise ValueError("RollingBuffer is empty")
        return self._sum / len(self._buffer)

    def variance(self, ddof: int = 0) -> float:
        if not self._buffer:
            raise ValueError("RollingBuffer is empty")
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        n = len(self._buffer)
        if ddof >= n:
            raise ValueError("ddof must be less than the number of observations")
        numerator = self._sum_sq - (self._sum * self._sum) / n
        denominator = n - ddof
        variance = max(numerator / denominator, 0.0)
        return variance

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(self.variance(ddof=ddof))

