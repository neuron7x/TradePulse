# SPDX-License-Identifier: MIT
from __future__ import annotations
import time, collections
from typing import Deque, Callable

class RollingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buf: Deque[float] = collections.deque(maxlen=size)

    def push(self, v: float):
        self.buf.append(v)

    def values(self):
        return list(self.buf)
