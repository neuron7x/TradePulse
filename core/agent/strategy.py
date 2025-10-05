# SPDX-License-Identifier: MIT
from __future__ import annotations
import math, random, time
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

@dataclass
class Strategy:
    name: str
    params: Dict[str, Any]
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def generate_mutation(self) -> "Strategy":
        new_params = {k: (v*(1+random.uniform(-0.2,0.2)) if isinstance(v,(int,float)) else v)
                      for k,v in self.params.items()}
        return Strategy(name=self.name + "_mut", params=new_params)

    def simulate_performance(self, data: Any) -> float:
        # stub (to be extended): user should plug actual backtest here
        self.score = random.uniform(-1.0, 2.0)
        return self.score

@dataclass
class PiAgent:
    strategy: Strategy

    def detect_instability(self, market_state: Dict[str, float]) -> bool:
        R = market_state.get("R", 0.0)
        dH = market_state.get("delta_H", 0.0)
        kappa = market_state.get("kappa_mean", 0.0)
        return (R > 0.75 and dH < 0 and kappa < 0)

    def mutate(self) -> "PiAgent":
        return PiAgent(strategy=self.strategy.generate_mutation())

    def repair(self) -> None:
        # reset bad parameters if any heuristic says so
        for k,v in list(self.strategy.params.items()):
            if isinstance(v, (int,float)) and (math.isnan(v) or abs(v)>1e6):
                self.strategy.params[k]=0.0

    def evaluate_and_adapt(self, market_state) -> str:
        if self.detect_instability(market_state):
            return "enter"
        elif market_state.get("phase_reversal", False):
            return "exit"
        return "hold"
