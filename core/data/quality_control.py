# SPDX-License-Identifier: MIT
"""Data quality gates for ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from core.data.validation import TimeSeriesValidationConfig, TimeSeriesValidationError, validate_timeseries_frame


class QualityGateError(TimeSeriesValidationError):
    """Raised when a batch violates a non-recoverable quality gate."""


class RangeCheck(BaseModel):
    """Declarative boundary constraints for numeric columns."""

    model_config = ConfigDict(frozen=True, strict=True)

    column: StrictStr = Field(..., min_length=1, description="Column subject to the range guard")
    min_value: StrictFloat | StrictInt | None = Field(
        default=None,
        description="Lower bound allowed for the column. ``None`` disables the guard.",
    )
    max_value: StrictFloat | StrictInt | None = Field(
        default=None,
        description="Upper bound allowed for the column. ``None`` disables the guard.",
    )
    inclusive_min: StrictBool = Field(default=True, description="Treat the lower bound as inclusive")
    inclusive_max: StrictBool = Field(default=True, description="Treat the upper bound as inclusive")

    @model_validator(mode="after")
    def _ensure_bounds(self) -> "RangeCheck":
        if self.min_value is None and self.max_value is None:
            raise QualityGateError("RangeCheck must define at least a minimum or maximum bound")
        if (
            self.min_value is not None
            and self.max_value is not None
            and float(self.min_value) > float(self.max_value)
        ):
            raise QualityGateError("RangeCheck minimum bound exceeds maximum bound")
        return self


class TemporalContract(BaseModel):
    """Guarantees about the temporal span of an ingestion batch."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    earliest: pd.Timestamp | None = Field(
        default=None,
        description="Earliest timestamp permitted in the batch",
    )
    latest: pd.Timestamp | None = Field(
        default=None,
        description="Latest timestamp permitted in the batch",
    )
    expected_start: pd.Timestamp | None = Field(
        default=None,
        description="Exact timestamp the batch must start at (within tolerance)",
    )
    expected_end: pd.Timestamp | None = Field(
        default=None,
        description="Exact timestamp the batch must end at (within tolerance)",
    )
    tolerance: pd.Timedelta = Field(
        default=pd.Timedelta(0),
        description="Tolerance applied when comparing expected start/end",
    )
    max_lag: pd.Timedelta | None = Field(
        default=None,
        description="Maximum allowed delay between now and the last timestamp",
    )

    @field_validator("earliest", "latest", "expected_start", "expected_end", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: object) -> pd.Timestamp | None:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise QualityGateError(f"Unable to parse timestamp value: {value!r}") from exc

    @field_validator("tolerance", mode="before")
    @classmethod
    def _coerce_timedelta(cls, value: object) -> pd.Timedelta:
        if isinstance(value, pd.Timedelta):
            return value
        if isinstance(value, timedelta):
            return pd.Timedelta(value)
        if value is None:
            return pd.Timedelta(0)
        try:
            return pd.Timedelta(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise QualityGateError(f"Unable to parse timedelta value: {value!r}") from exc

    @field_validator("max_lag", mode="before")
    @classmethod
    def _coerce_optional_timedelta(cls, value: object) -> pd.Timedelta | None:
        if value is None:
            return None
        if isinstance(value, pd.Timedelta):
            return value
        if isinstance(value, timedelta):
            return pd.Timedelta(value)
        try:
            return pd.Timedelta(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive path
            raise QualityGateError(f"Unable to parse timedelta value: {value!r}") from exc

    @model_validator(mode="after")
    def _validate_expectations(self) -> "TemporalContract":
        if self.expected_start and self.earliest and self.expected_start < self.earliest:
            raise QualityGateError("expected_start must not be before the earliest bound")
        if self.expected_end and self.latest and self.expected_end > self.latest:
            raise QualityGateError("expected_end must not exceed the latest bound")
        return self


class QualityGateConfig(BaseModel):
    """Composite quality gate configuration for ingestion payloads."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    schema: TimeSeriesValidationConfig = Field(..., description="Underlying schema contract")
    price_column: StrictStr = Field(
        default="close",
        min_length=1,
        description="Column used for anomaly detection thresholds",
    )
    anomaly_threshold: StrictFloat = Field(
        default=6.0,
        gt=0,
        description="Z-score at which a point is considered a spike",
    )
    anomaly_window: StrictInt = Field(
        default=20,
        ge=2,
        description="Window used to compute rolling statistics for z-score",
    )
    range_checks: Sequence[RangeCheck] = Field(
        default_factory=tuple,
        description="Per-column boundary guards",
    )
    temporal_contract: TemporalContract | None = Field(
        default=None,
        description="Optional temporal guarantees enforced on the batch",
    )
    max_quarantine_fraction: StrictFloat = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Maximum tolerated share of quarantined rows before blocking the batch",
    )

    @model_validator(mode="after")
    def _validate_price_column(self) -> "QualityGateConfig":
        available = {self.schema.timestamp_column, *(col.name for col in self.schema.value_columns)}
        if self.price_column not in available:
            raise QualityGateError(
                f"price_column '{self.price_column}' is not declared in the validation schema"
            )
        seen: set[str] = set()
        duplicates: set[str] = set()
        for check in self.range_checks:
            if check.column in seen:
                duplicates.add(check.column)
            seen.add(check.column)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise QualityGateError(f"Duplicate range checks defined for: {joined}")
        return self


@dataclass(slots=True)
class QualityReport:
    """Outcome of the quality gate validation."""

    clean: pd.DataFrame
    quarantined: pd.DataFrame
    duplicates: pd.DataFrame
    spikes: pd.DataFrame
    range_violations: Dict[str, pd.DataFrame] = field(default_factory=dict)
    contract_breaches: Tuple[str, ...] = field(default_factory=tuple)
    blocked: bool = False

    def raise_if_blocked(self) -> None:
        """Raise a :class:`QualityGateError` when the batch is unusable."""

        if not self.blocked:
            return
        reasons: List[str] = []
        if self.contract_breaches:
            reasons.extend(self.contract_breaches)
        if self.range_violations:
            for column, payload in self.range_violations.items():
                reasons.append(
                    f"{payload.shape[0]} rows in column '{column}' violate configured bounds"
                )
        if not reasons and not self.quarantined.empty:
            reasons.append("Quarantine ratio exceeded configured threshold")
        detail = "; ".join(reasons) or "Batch blocked by quality gate"
        raise QualityGateError(detail)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean().shift(1)
    std = rolling.std(ddof=0).shift(1)
    return (series - mean) / std.replace(0, np.nan)


def quarantine_anomalies(
    frame: pd.DataFrame,
    *,
    threshold: float,
    window: int,
    price_column: str,
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


def _apply_range_checks(
    frame: pd.DataFrame, checks: Iterable[RangeCheck], timestamp_col: str
) -> Dict[str, pd.DataFrame]:
    violations: Dict[str, pd.DataFrame] = {}
    for check in checks:
        if check.column not in frame.columns:
            raise QualityGateError(
                f"Range check column '{check.column}' is not present in the validated payload"
            )
        series = frame[check.column]
        mask = pd.Series(False, index=frame.index)
        if check.min_value is not None:
            min_value = float(check.min_value)
            mask |= series <= min_value if not check.inclusive_min else series < min_value
        if check.max_value is not None:
            max_value = float(check.max_value)
            mask |= series >= max_value if not check.inclusive_max else series > max_value
        mask &= series.notna()
        if mask.any():
            payload = frame.loc[mask].copy()
            violations[check.column] = payload.set_index(timestamp_col, drop=False)
    return violations


def _enforce_temporal_contract(
    timestamps: pd.Series, contract: TemporalContract
) -> Tuple[Tuple[str, ...], bool]:
    breaches: List[str] = []
    blocked = False
    if timestamps.empty:
        return tuple(breaches), blocked
    first = timestamps.iloc[0]
    last = timestamps.iloc[-1]
    tz = getattr(last, "tz", None)
    if contract.earliest is not None and first < contract.earliest:
        breaches.append(
            f"First timestamp {first} is earlier than allowed earliest {contract.earliest}"
        )
        blocked = True
    if contract.latest is not None and last > contract.latest:
        breaches.append(
            f"Last timestamp {last} exceeds allowed latest {contract.latest}"
        )
        blocked = True
    if contract.expected_start is not None:
        delta = abs(first - contract.expected_start)
        if delta > contract.tolerance:
            breaches.append(
                f"Batch starts at {first} but expected {contract.expected_start} ± {contract.tolerance}"
            )
            blocked = True
    if contract.expected_end is not None:
        delta = abs(last - contract.expected_end)
        if delta > contract.tolerance:
            breaches.append(
                f"Batch ends at {last} but expected {contract.expected_end} ± {contract.tolerance}"
            )
            blocked = True
    if contract.max_lag is not None:
        reference = pd.Timestamp.now(tz=tz)
        lag = reference - last
        if lag > contract.max_lag:
            breaches.append(
                f"Last timestamp is stale by {lag}, exceeding allowed lag {contract.max_lag}"
            )
            blocked = True
    return tuple(breaches), blocked


def validate_and_quarantine(frame: pd.DataFrame, gate: QualityGateConfig) -> QualityReport:
    """Validate a DataFrame and quarantine anomalies according to the configured gates."""

    config = gate.schema
    timestamp_col = config.timestamp_column
    duplicates = frame[frame[timestamp_col].duplicated(keep=False)]
    working = frame.drop_duplicates(subset=timestamp_col, keep="first").copy()
    for column in config.value_columns:
        if column.dtype:
            working[column.name] = working[column.name].astype(column.dtype, copy=False)
    validated = validate_timeseries_frame(working, config)

    buckets = quarantine_anomalies(
        validated.set_index(timestamp_col),
        threshold=float(gate.anomaly_threshold),
        window=int(gate.anomaly_window),
        price_column=gate.price_column,
    )

    clean = buckets["clean"].reset_index()
    quarantined = pd.concat([buckets["spikes"], buckets["duplicates"]]).drop_duplicates()
    quarantined = pd.concat(
        [quarantined, duplicates.set_index(timestamp_col)], axis=0
    ).drop_duplicates()
    quarantined = quarantined.reset_index()

    range_violations = _apply_range_checks(validated, gate.range_checks, timestamp_col)
    if range_violations:
        violation_index = pd.Index([])
        for payload in range_violations.values():
            violation_index = violation_index.union(payload.index)
        combined_range = (
            pd.concat(range_violations.values(), axis=0)
            .reset_index(drop=True)
            .loc[:, validated.columns]
            .drop_duplicates()
        )
        if not clean.empty:
            clean = (
                clean.set_index(timestamp_col)
                .drop(index=violation_index, errors="ignore")
                .reset_index()
            )
        combined_range_indexed = combined_range.set_index(timestamp_col)
        if not quarantined.empty:
            quarantined = (
                pd.concat([quarantined.set_index(timestamp_col), combined_range_indexed], axis=0)
                .drop_duplicates()
                .reset_index()
            )
        else:
            quarantined = combined_range_indexed.reset_index()
    contract_breaches: Tuple[str, ...] = tuple()
    contract_block = False
    if gate.temporal_contract is not None:
        contract_breaches, contract_block = _enforce_temporal_contract(
            validated[timestamp_col], gate.temporal_contract
        )

    blocked = contract_block or bool(range_violations)
    total_rows = max(validated.shape[0], 1)
    if not blocked and not quarantined.empty:
        ratio = quarantined.shape[0] / total_rows
        if ratio > float(gate.max_quarantine_fraction):
            blocked = True

    report = QualityReport(
        clean=clean,
        quarantined=quarantined,
        duplicates=duplicates.reset_index(drop=True),
        spikes=buckets["spikes"].reset_index(),
        range_violations=range_violations,
        contract_breaches=contract_breaches,
        blocked=blocked,
    )
    return report


__all__ = [
    "QualityGateConfig",
    "QualityGateError",
    "QualityReport",
    "RangeCheck",
    "TemporalContract",
    "quarantine_anomalies",
    "validate_and_quarantine",
]

