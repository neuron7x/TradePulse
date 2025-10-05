# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" in df: df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df = df.sort_values("ts")
    df = df.drop_duplicates()
    df = df.interpolate()
    return df

def scale_series(x, method="zscore"):
    import numpy as np
    x = np.asarray(x, dtype=float)
    if method=="zscore":
        mu, sd = x.mean(), x.std() + 1e-12
        return (x-mu)/sd
    return x
