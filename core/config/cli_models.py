"""Lightweight Pydantic models for TradePulse CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .postgres import ensure_secure_postgres_uri, is_postgres_uri

__all__ = [
    "CatalogConfig",
    "DataSourceConfig",
    "ExperimentAnalyticsConfig",
    "ExperimentConfig",
    "ExperimentDataConfig",
    "ExperimentTrackingConfig",
    "ExecutionConfig",
    "FeatureFrameSourceConfig",
    "FeatureParityConfig",
    "FeatureParitySpecConfig",
    "IngestConfig",
    "OptimizeConfig",
    "PostgresTLSConfig",
    "ReportConfig",
    "StrategyConfig",
    "TradePulseBaseConfig",
    "VersioningConfig",
]


class VersioningConfig(BaseModel):
    """Configuration describing how artifacts should be versioned."""

    backend: Literal["none", "dvc", "lakefs"] = "none"
    repo_path: Optional[Path] = None
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_repo(self) -> "VersioningConfig":
        if self.backend != "none" and self.repo_path is None:
            msg = "repo_path is required when using dvc or lakefs backends"
            raise ValueError(msg)
        return self


class PostgresTLSConfig(BaseModel):
    """TLS material required for PostgreSQL client authentication."""

    ca_file: Path
    cert_file: Path
    key_file: Path


class ExperimentDataConfig(BaseModel):
    """Dataset location for experiment-oriented jobs."""

    price_csv: Path
    price_column: str


class ExperimentAnalyticsConfig(BaseModel):
    """Numeric parameters for experiment analytics windows."""

    window: int
    bins: int
    delta: float


class ExperimentTrackingConfig(BaseModel):
    """Where to persist experiment tracking artifacts."""

    enabled: bool = True
    base_dir: Path


class ExperimentConfig(BaseModel):
    """Hydra experiment configuration with strict database validation."""

    name: str
    db_uri: str
    db_tls: PostgresTLSConfig | None = None
    debug: bool = False
    log_level: str = "INFO"
    random_seed: int = 0
    data: ExperimentDataConfig
    analytics: ExperimentAnalyticsConfig
    tracking: ExperimentTrackingConfig

    @model_validator(mode="after")
    def _validate_database_security(self) -> "ExperimentConfig":
        ensure_secure_postgres_uri(self.db_uri)
        if is_postgres_uri(self.db_uri) and self.db_tls is None:
            msg = "db_tls must be provided when using a PostgreSQL database"
            raise ValueError(msg)
        return self


class CatalogConfig(BaseModel):
    """Simple file-backed catalog configuration."""

    path: Path = Field(default=Path("data/feature_catalog.json"))


class DataSourceConfig(BaseModel):
    """Location of input data for CLI jobs."""

    kind: Literal["csv", "parquet"] = "csv"
    path: Path
    timestamp_field: str = "timestamp"
    value_field: str = "price"


class StrategyConfig(BaseModel):
    """Describes a callable that returns trading signals."""

    entrypoint: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_entrypoint(self) -> "StrategyConfig":
        if ":" not in self.entrypoint:
            raise ValueError("entrypoint must be of the form 'module:function'")
        return self


class TradePulseBaseConfig(BaseModel):
    """Common metadata shared by CLI configurations."""

    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestConfig(TradePulseBaseConfig):
    """Configuration driving the ingest command."""

    source: DataSourceConfig
    destination: Path
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class ExecutionConfig(BaseModel):
    """Execution parameters for the backtest command."""

    starting_cash: float = 100_000.0
    fee_bps: float = 0.0


class BacktestConfig(TradePulseBaseConfig):
    """Configuration for running a simple vectorized backtest."""

    data: DataSourceConfig
    strategy: StrategyConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    results_path: Path = Field(default=Path("reports/backtest.json"))
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class OptimizeConfig(TradePulseBaseConfig):
    """Configuration for parameter search via grid search."""

    objective: str
    search_space: Dict[str, Iterable[Any]]
    results_path: Path = Field(default=Path("reports/optimize.json"))
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @model_validator(mode="after")
    def _validate_objective(self) -> "OptimizeConfig":
        if ":" not in self.objective:
            raise ValueError("objective must be of the form 'module:function'")
        return self


class ExecConfig(TradePulseBaseConfig):
    """Configuration for running a real-time signal evaluation."""

    data: DataSourceConfig
    strategy: StrategyConfig
    results_path: Path = Field(default=Path("reports/exec.json"))
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class ReportConfig(TradePulseBaseConfig):
    """Configuration for aggregating CLI outputs into a report."""

    inputs: List[Path]
    output_path: Path
    html_output_path: Path | None = None
    pdf_output_path: Path | None = None
    template: Optional[Path] = None
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class FeatureFrameSourceConfig(BaseModel):
    """Location of offline feature snapshots used for parity checks."""

    path: Path
    format: Literal["auto", "csv", "parquet"] = "auto"


class FeatureParitySpecConfig(BaseModel):
    """Declarative parity expectations for a feature view."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_view: str
    entity_columns: Tuple[str, ...] = ("entity_id",)
    timestamp_column: str = "ts"
    timestamp_granularity: pd.Timedelta | str | None = None
    numeric_tolerance: float | None = 0.0
    max_clock_skew: pd.Timedelta | str | None = "0s"
    allow_schema_evolution: bool = False
    value_columns: Tuple[str, ...] | None = None

    @field_validator("timestamp_granularity", "max_clock_skew", mode="before")
    def _parse_timedelta(cls, value: object) -> pd.Timedelta | None:
        if value is None or isinstance(value, pd.Timedelta):
            return value
        if isinstance(value, str) and value.strip().lower() in {"", "none"}:
            return None
        try:
            return pd.Timedelta(value)  # type: ignore[arg-type]
        except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
            raise ValueError("timedelta fields must be pandas-compatible strings") from exc

    @model_validator(mode="after")
    def _validate_columns(self) -> "FeatureParitySpecConfig":
        if not self.entity_columns:
            raise ValueError("entity_columns must define at least one column")
        return self


class FeatureParityConfig(TradePulseBaseConfig):
    """Top-level configuration driving the feature parity CLI command."""

    offline: FeatureFrameSourceConfig
    online_store: Path = Field(default=Path("data/online_features"))
    mode: Literal["append", "overwrite"] = "append"
    spec: FeatureParitySpecConfig

    @model_validator(mode="after")
    def _validate_mode(self) -> "FeatureParityConfig":
        if self.mode not in {"append", "overwrite"}:
            raise ValueError("mode must be either 'append' or 'overwrite'")
        return self
