"""Data quality gates for ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from core.data.validation import TimeSeriesValidationConfig, validate_timeseries_frame


@dataclass
class QualityReport:
    """Outcome of the quality gate validation."""

    clean: pd.DataFrame
    quarantined: pd.DataFrame
    duplicates: pd.DataFrame
    spikes: pd.DataFrame


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean().shift(1)
    std = rolling.std(ddof=0).shift(1)
    return (series - mean) / std.replace(0, np.nan)


def quarantine_anomalies(
    frame: pd.DataFrame,
    *,
    threshold: float = 6.0,
    window: int = 20,
    price_column: str = "close",
) -> Dict[str, pd.DataFrame]:
    """Split the frame into clean rows and anomalies based on z-score."""

    if frame.empty:
        return {"clean": frame, "spikes": frame, "duplicates": frame}
    duplicates = frame[frame.index.duplicated(keep=False)]
    deduped = frame[~frame.index.duplicated(keep="first")]
    scores = _zscore(deduped[price_column], window)
    spikes = deduped[np.abs(scores) > threshold]
    clean = deduped.drop(spikes.index, errors="ignore")
    return {"clean": clean, "spikes": spikes, "duplicates": duplicates}


def validate_and_quarantine(
    frame: pd.DataFrame,
    config: TimeSeriesValidationConfig,
    *,
    threshold: float = 6.0,
    window: int = 20,
    price_column: str = "close",
) -> QualityReport:
    """Validate a DataFrame and quarantine anomalies."""

    timestamp_col = config.timestamp_column
    duplicates = frame[frame[timestamp_col].duplicated(keep=False)]
    working = frame.drop_duplicates(subset=timestamp_col, keep="first").copy()
    for column in config.value_columns:
        if column.dtype:
            working[column.name] = working[column.name].astype(column.dtype, copy=False)
    validated = validate_timeseries_frame(working, config)
    buckets = quarantine_anomalies(
        validated.set_index(timestamp_col),
        threshold=threshold,
        window=window,
        price_column=price_column,
    )
    clean = buckets["clean"].reset_index()
    quarantined = pd.concat([buckets["spikes"], buckets["duplicates"]]).drop_duplicates()
    return QualityReport(
        clean=clean,
        quarantined=pd.concat([quarantined, duplicates.set_index(timestamp_col)]).reset_index(),
        duplicates=duplicates.reset_index(drop=True),
        spikes=buckets["spikes"].reset_index(),
    )


__all__ = ["QualityReport", "quarantine_anomalies", "validate_and_quarantine"]

