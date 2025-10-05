# SPDX-License-Identifier: MIT
from __future__ import annotations
import math, random
from typing import Dict, List

class EpsilonGreedy:
    def __init__(self, arms: List[str], epsilon: float = 0.1):
        self.Q = {a: 0.0 for a in arms}
        self.N = {a: 0 for a in arms}
        self.epsilon = epsilon

    def select(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(list(self.Q.keys()))
        return max(self.Q, key=self.Q.get)

    def update(self, arm: str, reward: float):
        self.N[arm] += 1
        n = self.N[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n

class UCB1:
    def __init__(self, arms: List[str]):
        self.Q = {a: 0.0 for a in arms}
        self.N = {a: 0 for a in arms}
        self.t = 0

    def select(self) -> str:
        self.t += 1
        def ucb(a):
            n = self.N[a]
            if n == 0: return float("inf")
            return self.Q[a] + math.sqrt(2*math.log(self.t)/n)
        return max(self.Q.keys(), key=ucb)

    def update(self, arm: str, reward: float):
        self.N[arm] += 1
        n = self.N[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n
