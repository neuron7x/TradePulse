"""Lightweight Pydantic models for TradePulse CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .postgres import ensure_secure_postgres_uri, is_postgres_uri

__all__ = [
    "CatalogConfig",
    "DataSourceConfig",
    "ExperimentAnalyticsConfig",
    "ExperimentConfig",
    "ExperimentDataConfig",
    "ExperimentTrackingConfig",
    "ExecutionConfig",
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
