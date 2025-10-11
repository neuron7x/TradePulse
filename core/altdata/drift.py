"""Distribution drift monitoring utilities for alternative data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DriftAssessment:
    """Summary of a drift evaluation."""

    metric: str
    value: float
    threshold: float
    drifted: bool
    details: dict[str, float]


class DistributionDriftMonitor:
    """Assess population drift for alternative data feature streams."""

    def __init__(self, *, method: str = "psi", threshold: float = 0.2, bins: int = 10) -> None:
        self._method = method.lower()
        if self._method not in {"psi", "ks"}:
            raise ValueError("method must be either 'psi' or 'ks'")
        self._threshold = float(threshold)
        self._bins = max(3, int(bins))

    def _ensure_series(self, values: Iterable[float]) -> pd.Series:
        series = pd.Series(list(values), dtype=float).dropna()
        if series.empty:
            raise ValueError("Input series must contain at least one value")
        return series

    def _psi(self, reference: pd.Series, current: pd.Series) -> DriftAssessment:
        quantiles = np.linspace(0, 1, self._bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        if len(edges) < 2:
            return DriftAssessment("psi", 0.0, self._threshold, False, {"bins": len(edges)})
        ref_hist, _ = np.histogram(reference, bins=edges)
        cur_hist, _ = np.histogram(current, bins=edges)
        ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
        cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
        psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
        return DriftAssessment("psi", psi, self._threshold, psi >= self._threshold, {"bins": float(len(edges) - 1)})

    def _ks(self, reference: pd.Series, current: pd.Series) -> DriftAssessment:
        statistic, pvalue = stats.ks_2samp(reference, current)
        drifted = pvalue < (1 - self._threshold)
        return DriftAssessment("ks", float(statistic), self._threshold, drifted, {"pvalue": float(pvalue)})

    def assess(self, reference: Iterable[float], current: Iterable[float]) -> DriftAssessment:
        """Evaluate drift between reference and current samples."""

        ref_series = self._ensure_series(reference)
        cur_series = self._ensure_series(current)
        if self._method == "psi":
            return self._psi(ref_series, cur_series)
        return self._ks(ref_series, cur_series)


__all__ = ["DistributionDriftMonitor", "DriftAssessment"]
