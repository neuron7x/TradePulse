"""Daily FinOps cost attribution reporting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_RESOURCE_COLUMNS: tuple[str, ...] = ("cpu_cost", "gpu_cost", "io_cost")


def _ensure_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Cast the requested columns to floats, filling missing entries with zero."""

    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return frame


def _format_segment(row: Mapping[str, Any], dimensions: Sequence[str]) -> str:
    if not dimensions:
        return "global"
    parts: list[str] = []
    for name in dimensions:
        value = row.get(name, "<unknown>")
        if pd.isna(value):
            value = "<missing>"
        parts.append(f"{name}={value}")
    return ", ".join(parts)


def _dataframe_to_markdown(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No data available._"

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, pd.Timestamp):
                cells.append(value.strftime("%Y-%m-%d"))
            elif isinstance(value, (float, np.floating)):
                if math.isnan(float(value)):
                    cells.append("—")
                else:
                    cells.append(f"{float(value):.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass(slots=True)
class DailyCostReport:
    """Container holding derived FinOps cost analytics."""

    daily_totals: pd.DataFrame
    confidence_intervals: pd.DataFrame
    regression_signals: list[dict[str, Any]]
    metadata: dict[str, Any]
    resources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the report."""

        def _normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
            normalised: dict[str, Any] = {}
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    normalised[key] = value.strftime("%Y-%m-%d")
                elif isinstance(value, (np.floating, float)):
                    if math.isnan(float(value)):
                        normalised[key] = None
                    else:
                        normalised[key] = float(value)
                elif isinstance(value, (np.integer, int)):
                    normalised[key] = int(value)
                elif value is None or (isinstance(value, float) and math.isnan(value)):
                    normalised[key] = None
                else:
                    normalised[key] = value
            return normalised

        daily_payload = [_normalise_record(record) for record in self.daily_totals.to_dict(orient="records")]
        confidence_payload = [_normalise_record(record) for record in self.confidence_intervals.to_dict(orient="records")]
        signals_payload: list[dict[str, Any]] = []
        for signal in self.regression_signals:
            normalised = {}
            for key, value in signal.items():
                if isinstance(value, (np.floating, float)):
                    normalised[key] = float(value)
                elif isinstance(value, (np.integer, int)):
                    normalised[key] = int(value)
                else:
                    normalised[key] = value
            signals_payload.append(normalised)
        return {
            "daily_totals": daily_payload,
            "confidence_intervals": confidence_payload,
            "regression_signals": signals_payload,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Render a lightweight Markdown summary of the report."""

        lines = ["# Daily Cost Report"]
        generated = self.metadata.get("generated_at")
        if generated:
            lines.append(f"_Generated at {generated}_")
        lines.append("")

        latest_date = self.daily_totals["date"].max()
        latest_mask = self.daily_totals["date"] == latest_date
        display_columns = ["segment", *self.resources, "total_cost"]
        latest_view = self.daily_totals.loc[latest_mask, display_columns]
        lines.append(f"## Totals for {latest_date.strftime('%Y-%m-%d')}")
        lines.append(_dataframe_to_markdown(latest_view, display_columns))
        lines.append("")

        latest_confidence = (
            self.confidence_intervals.sort_values(["segment", "resource", "date"])
            .groupby(["segment", "resource"], as_index=False)
            .tail(1)
        )
        if not latest_confidence.empty:
            confidence_columns = ["segment", "resource", "value", "mean", "lower", "upper"]
            lines.append("## Confidence Bands")
            lines.append(_dataframe_to_markdown(latest_confidence[confidence_columns], confidence_columns))
            lines.append("")

        lines.append("## Regression Signals")
        if not self.regression_signals:
            lines.append("No significant regression signals detected.")
        else:
            frame = pd.DataFrame(self.regression_signals)
            frame = frame.sort_values(["severity", "segment", "resource"], ascending=[False, True, True])
            signal_columns = [
                "segment",
                "resource",
                "severity",
                "latest_value",
                "baseline_mean",
                "slope",
                "trend_strength",
                "z_score",
            ]
            # Ensure optional columns exist
            for column in signal_columns:
                if column not in frame.columns:
                    frame[column] = np.nan
            lines.append(_dataframe_to_markdown(frame[signal_columns], signal_columns))
        return "\n".join(lines)


def build_daily_cost_report(
    records: pd.DataFrame,
    *,
    segments: Sequence[str] | None = None,
    confidence: float = 0.95,
    baseline_window: int = 7,
    trend_threshold: float = 0.15,
    zscore_threshold: float = 1.5,
) -> DailyCostReport:
    """Generate a structured FinOps cost report from raw records."""

    if records.empty:
        raise ValueError("Cost records are empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence {confidence!r} must lie in the open interval (0, 1)"
        )
    if baseline_window < 2:
        raise ValueError(
            f"baseline_window {baseline_window!r} must be at least 2"
        )
    if trend_threshold <= 0.0:
        raise ValueError(f"trend_threshold {trend_threshold!r} must be positive")
    if zscore_threshold <= 0.0:
        raise ValueError(
            f"zscore_threshold {zscore_threshold!r} must be positive"
        )

    frame = records.copy()

    timestamp_field = "timestamp"
    if timestamp_field not in frame.columns:
        raise ValueError("Cost records must include a 'timestamp' column")

    timestamps = pd.to_datetime(frame[timestamp_field], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Encountered invalid timestamps in cost records")
    frame["date"] = timestamps.dt.tz_localize(None).dt.normalize()

    frame = _ensure_numeric_columns(frame, _RESOURCE_COLUMNS)
    frame["total_cost"] = frame[list(_RESOURCE_COLUMNS)].sum(axis=1)

    dimensions = [col for col in (segments or []) if col in frame.columns]
    group_columns = ["date", *dimensions]
    grouped = (
        frame.groupby(group_columns, dropna=False)[[*_RESOURCE_COLUMNS, "total_cost"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if dimensions:
        grouped["segment"] = grouped.apply(lambda row: _format_segment(row, dimensions), axis=1)
    else:
        grouped["segment"] = "global"

    # Append global aggregates so dashboards can show fleet-wide totals.
    if dimensions:
        global_totals = (
            grouped.groupby("date")[[*_RESOURCE_COLUMNS, "total_cost"]]
            .sum()
            .reset_index()
            .assign(**{dimension: "<all>" for dimension in dimensions})
        )
        global_totals["segment"] = "global"
        grouped = pd.concat([grouped, global_totals], ignore_index=True, sort=False)
    grouped = grouped.sort_values(["date", "segment"]).reset_index(drop=True)

    resources = (*_RESOURCE_COLUMNS,)
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    confidence_rows: list[dict[str, Any]] = []
    for segment_key, segment_df in grouped.groupby("segment", sort=False):
        segment_df = segment_df.sort_values("date")
        for resource in [*_RESOURCE_COLUMNS, "total_cost"]:
            series = segment_df[resource].astype(float)
            rolling = series.rolling(window=baseline_window, min_periods=1)
            means = rolling.mean()
            stds = rolling.std(ddof=1)
            counts = rolling.count()
            for date, value, mean, std, count in zip(segment_df["date"], series, means, stds, counts):
                if count < 2 or math.isnan(std):
                    lower = float("nan")
                    upper = float("nan")
                else:
                    margin = z_value * std / math.sqrt(count)
                    lower = float(mean - margin)
                    upper = float(mean + margin)
                confidence_rows.append(
                    {
                        "date": date,
                        "segment": segment_key,
                        "resource": resource,
                        "value": float(value),
                        "mean": float(mean),
                        "lower": lower,
                        "upper": upper,
                    }
                )
    confidence_frame = pd.DataFrame(confidence_rows)

    regression_signals: list[dict[str, Any]] = []
    for segment_key, segment_df in grouped.groupby("segment", sort=False):
        segment_df = segment_df.sort_values("date")
        if len(segment_df) < 3:
            continue
        for resource in [*_RESOURCE_COLUMNS, "total_cost"]:
            values = segment_df[resource].astype(float).to_numpy()
            window = min(baseline_window, values.size)
            tail = values[-window:]
            slope = 0.0
            if np.allclose(tail, tail[0]):
                trend_strength = 0.0
            else:
                x = np.arange(window, dtype=float)
                slope, _ = np.polyfit(x, tail, 1)
                trend_strength = slope / (float(np.mean(tail)) + 1e-9)
            mean = float(np.mean(tail))
            std = float(np.std(tail, ddof=1)) if window > 1 else 0.0
            latest = float(tail[-1])
            z_score = 0.0 if std == 0.0 else (latest - mean) / std
            severity = "info"
            if z_score >= zscore_threshold or trend_strength >= trend_threshold * 1.5:
                severity = "critical"
            elif z_score >= zscore_threshold * 0.75 or trend_strength >= trend_threshold:
                severity = "warning"
            regression_signals.append(
                {
                    "segment": segment_key,
                    "resource": resource,
                    "latest_value": latest,
                    "baseline_mean": mean,
                    "slope": slope,
                    "trend_strength": trend_strength,
                    "z_score": z_score,
                    "window": window,
                    "severity": severity,
                }
            )

    # Filter out purely informational signals for noise reduction.
    regression_signals = [signal for signal in regression_signals if signal["severity"] != "info"]

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_level": confidence,
        "baseline_window": baseline_window,
        "trend_threshold": trend_threshold,
        "zscore_threshold": zscore_threshold,
        "segments": list(dimensions),
        "record_count": int(len(frame)),
        "rows": int(len(grouped)),
        "latest_date": grouped["date"].max().strftime("%Y-%m-%d"),
    }

    return DailyCostReport(
        daily_totals=grouped,
        confidence_intervals=confidence_frame,
        regression_signals=regression_signals,
        metadata=metadata,
        resources=resources,
    )


__all__ = ["DailyCostReport", "build_daily_cost_report"]
