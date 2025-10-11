# SPDX-License-Identifier: MIT
"""Lightweight memory pooling utilities used by performance critical paths."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import MutableMapping

import numpy as np


class ArrayPool:
    """Reusable pool of NumPy arrays for zero-copy indicator pipelines."""

    def __init__(self, dtype: np.dtype | str = np.float32) -> None:
        self.dtype = np.dtype(dtype)
        self._pool: MutableMapping[tuple[tuple[int, ...], np.dtype], list[np.ndarray]] = defaultdict(list)

    def acquire(self, shape: Iterable[int], *, dtype: np.dtype | str | None = None) -> np.ndarray:
        requested_dtype = np.dtype(dtype) if dtype is not None else self.dtype
        key = (tuple(int(s) for s in shape), requested_dtype)
        bucket = self._pool.get(key)
        if bucket:
            return bucket.pop()
        return np.empty(key[0], dtype=requested_dtype)

    def release(self, array: np.ndarray) -> None:
        key = (tuple(int(s) for s in array.shape), array.dtype)
        self._pool[key].append(array)

    def clear(self) -> None:
        self._pool.clear()


__all__ = ["ArrayPool"]
