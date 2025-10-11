"""Objective helpers for parameter optimisation routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.agent.strategy import Strategy

__all__ = ["mean_reversion_objective"]


def mean_reversion_objective(
    parameters: Dict[str, Any],
    *,
    data_path: str,
    price_column: str = "close",
) -> float:
    """Evaluate :class:`core.agent.strategy.Strategy` on a dataset."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    frame = pd.read_csv(path)
    if price_column not in frame.columns:
        raise ValueError(f"Column '{price_column}' not present in {data_path}")

    strategy = Strategy(name="mean_reversion", params=dict(parameters))
    return strategy.simulate_performance(frame[[price_column]].rename(columns={price_column: "close"}))
