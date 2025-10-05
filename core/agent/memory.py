# SPDX-License-Identifier: MIT
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

Addr = Tuple[float,float,float,float,float]  # (R, ΔH, κ̄, H, ISM)

@dataclass
class StrategyRecord:
    name: str
    addr: Addr
    score: float
    ts: float

class StrategyMemory:
    def __init__(self, decay_lambda: float = 1e-6):
        self.records: List[StrategyRecord] = []
        self.lmb = decay_lambda

    def add(self, name: str, addr: Addr, score: float):
        self.records.append(StrategyRecord(name, addr, score, time.time()))

    def freshness(self, rec: StrategyRecord) -> float:
        age = time.time() - rec.ts
        return math.exp(-self.lmb * age) * rec.score

    def topk(self, k=5) -> List[StrategyRecord]:
        return sorted(self.records, key=self.freshness, reverse=True)[:k]

    def cleanup(self, min_score: float = 0.0):
        self.records = [r for r in self.records if self.freshness(r) > min_score]
