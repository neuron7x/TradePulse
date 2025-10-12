"""Lightweight Pydantic models for TradePulse CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CatalogConfig",
    "DataSourceConfig",
    "CostReportConfig",
    "CostSourceConfig",
    "ExecutionConfig",
    "IngestConfig",
    "OptimizeConfig",
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


class CatalogConfig(BaseModel):
    """Simple file-backed catalog configuration."""

    path: Path = Field(default=Path("data/feature_catalog.json"))


class DataSourceConfig(BaseModel):
    """Location of input data for CLI jobs."""

    kind: Literal["csv", "parquet"] = "csv"
    path: Path
    timestamp_field: str = "timestamp"
    value_field: str = "price"


class CostSourceConfig(BaseModel):
    """Configuration describing where FinOps cost records live."""

    kind: Literal["csv", "parquet"] = "csv"
    path: Path
    timestamp_field: str = "timestamp"
    dimensions: List[str] = Field(default_factory=lambda: ["team", "environment", "service"])
    cpu_field: str = "cpu_cost"
    gpu_field: str = "gpu_cost"
    io_field: str = "io_cost"

    @model_validator(mode="after")
    def _validate_fields(self) -> "CostSourceConfig":
        resources = {self.cpu_field, self.gpu_field, self.io_field}
        if len(resources) != 3:
            raise ValueError("Resource field names must be distinct")
        return self


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


class CostReportConfig(TradePulseBaseConfig):
    """Configuration for FinOps daily cost reporting."""

    source: CostSourceConfig
    output_path: Path = Field(default=Path("reports/finops/daily_cost_report.json"))
    markdown_output_path: Path | None = Field(default=Path("reports/finops/daily_cost_report.md"))
    baseline_window: int = 7
    confidence_level: float = 0.95
    trend_threshold: float = 0.15
    zscore_threshold: float = 1.5
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @model_validator(mode="after")
    def _validate_parameters(self) -> "CostReportConfig":
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in the open interval (0, 1)")
        if self.baseline_window < 3:
            raise ValueError("baseline_window must be at least 3")
        if self.trend_threshold <= 0.0:
            raise ValueError("trend_threshold must be positive")
        if self.zscore_threshold <= 0.0:
            raise ValueError("zscore_threshold must be positive")
        return self
