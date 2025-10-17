"""Data loading utilities for CSV/Parquet sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from core.backtest import BarData
from core.utils_time import ensure_utc


@dataclass(slots=True)
class CSVLoaderConfig:
    path: Path
    tz: str = "UTC"


def load_csv_bars(cfg: CSVLoaderConfig) -> List[BarData]:
    df = pd.read_csv(cfg.path, parse_dates=["timestamp"])
    bars: List[BarData] = []
    for row in df.itertuples(index=False):
        close = float(getattr(row, "close"))
        open_ = float(getattr(row, "open", close))
        high = float(getattr(row, "high", max(open_, close)))
        low = float(getattr(row, "low", min(open_, close)))
        bars.append(
            BarData(
                timestamp=ensure_utc(getattr(row, "timestamp")),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=float(getattr(row, "volume", 0.0)),
                vwap=float(getattr(row, "vwap", close)),
                atr=float(getattr(row, "atr", 0.0)) or None,
            )
        )
    return bars

