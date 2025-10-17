# SPDX-License-Identifier: MIT
# concrete adapter for SumPort
from src.ports.ports import SumPort


class LocalSum(SumPort):
    def sum(self, a: int, b: int) -> int:
        return a + b
