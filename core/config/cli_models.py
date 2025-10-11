"""Pydantic models for the TradePulse CLI configuration workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

__all__ = [
    "CatalogConfig",
    "DataSourceConfig",
    "ExecutionConfig",
    "IngestConfig",
    "OptimizeConfig",
    "ReportConfig",
    "StrategyConfig",
    "TradePulseBaseConfig",
    "VersioningConfig",
]


def _resolve_path(value: Any, info: ValidationInfo) -> Path:
    if value is None:
        return value
    path = Path(value)
    base_path = None
    if info.context is not None:
        base_path = info.context.get("base_path")
    if base_path and not path.is_absolute():
        path = (Path(base_path) / path).resolve()
    return path


class VersioningConfig(BaseModel):
    """Version control configuration for data artifacts."""

    backend: Literal["none", "dvc", "lakefs"] = "none"
    repo_path: Optional[Path] = None
    remote: Optional[str] = None
    branch: Optional[str] = None
    commit_message: Optional[str] = Field(default=None, description="Optional commit message override")

    @field_validator("repo_path", mode="before")
    @classmethod
    def _resolve_repo_path(cls, value: Any, info: ValidationInfo) -> Path | None:
        if value in (None, ""):
            return None
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _validate_backend(self) -> "VersioningConfig":
        if self.backend in {"dvc", "lakefs"} and self.repo_path is None:
            raise ValueError("repo_path must be provided when using dvc or lakefs backends")
        return self


class CatalogConfig(BaseModel):
    """Feature catalog storage configuration."""

    path: Path = Field(default=Path("data/feature_catalog.json"))

    @field_validator("path", mode="before")
    @classmethod
    def _resolve_path_field(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)


class DataSourceConfig(BaseModel):
    """Data source definition used across CLI commands."""

    kind: Literal["csv", "parquet", "api", "stream"] = "csv"
    path: Optional[Path] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _resolve_data_path(cls, value: Any, info: ValidationInfo) -> Path | None:
        if value in (None, ""):
            return None
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _check_path(self) -> "DataSourceConfig":
        if self.kind in {"csv", "parquet"} and self.path is None:
            raise ValueError("path must be provided for csv and parquet data sources")
        return self


class StrategyConfig(BaseModel):
    """Encapsulates the strategy callable used by backtest/exec/optimize flows."""

    entrypoint: str = Field(..., min_length=3, description="<module>:<callable> path")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_entrypoint(self) -> "StrategyConfig":
        if ":" not in self.entrypoint:
            raise ValueError("entrypoint must be in '<module>:<callable>' format")
        return self


class TradePulseBaseConfig(BaseModel):
    """Base configuration shared across CLI commands."""

    name: str = Field(..., min_length=3)
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = Field(default="v1", description="Config version label")


class IngestConfig(TradePulseBaseConfig):
    """Configuration for the ingest sub-command."""

    source: DataSourceConfig
    destination: Path = Field(default=Path("data/processed/ingested.parquet"))
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    features: Dict[str, Any] = Field(default_factory=dict)
    lineage: List[str] = Field(default_factory=list)

    @field_validator("destination", mode="before")
    @classmethod
    def _resolve_destination(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _validate_destination(self) -> "IngestConfig":
        if self.destination.suffix == "":
            raise ValueError("destination must include a file extension (e.g. .csv or .parquet)")
        return self


class ExecutionConfig(BaseModel):
    initial_capital: float = 100_000.0
    fee_bps: float = 0.0
    chunk_size: Optional[int] = None
    latency: Dict[str, int] = Field(default_factory=dict)


class BacktestConfig(TradePulseBaseConfig):
    """Configuration for the backtest sub-command."""

    data: DataSourceConfig
    strategy: StrategyConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    results_path: Path = Field(default=Path("reports/backtests/result.json"))
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)

    @field_validator("results_path", mode="before")
    @classmethod
    def _resolve_results_path(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _validate_results_path(self) -> "BacktestConfig":
        if self.results_path.suffix.lower() not in {".json", ".parquet", ".csv"}:
            raise ValueError("results_path must end with .json, .csv or .parquet")
        return self


class OptimizeConfig(TradePulseBaseConfig):
    """Configuration for the optimize sub-command."""

    objective: str = Field(..., min_length=3)
    direction: Literal["maximize", "minimize"] = "maximize"
    search_space: Dict[str, Iterable[Any]]
    max_trials: Optional[int] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    results_path: Path = Field(default=Path("reports/optimization/result.json"))
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @field_validator("results_path", mode="before")
    @classmethod
    def _resolve_opt_results(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _validate_objective(self) -> "OptimizeConfig":
        if ":" not in self.objective:
            raise ValueError("objective must be in '<module>:<callable>' format")
        return self


class ExecConfig(TradePulseBaseConfig):
    """Configuration for the exec sub-command."""

    strategy: StrategyConfig
    broker: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)


class ReportConfig(TradePulseBaseConfig):
    """Configuration for the report sub-command."""

    inputs: List[Path]
    output_path: Path
    template: Optional[Path] = None
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @field_validator("inputs", mode="before")
    @classmethod
    def _resolve_inputs(cls, value: Any, info: ValidationInfo) -> List[Path]:
        paths: List[Path] = []
        base_path = Path(info.context.get("base_path")) if info.context and "base_path" in info.context else None
        for entry in value or []:
            path = Path(entry)
            if base_path and not path.is_absolute():
                path = (base_path / path).resolve()
            paths.append(path)
        return paths

    @field_validator("output_path", mode="before")
    @classmethod
    def _resolve_output(cls, value: Any, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)

    @field_validator("template", mode="before")
    @classmethod
    def _resolve_template(cls, value: Any, info: ValidationInfo) -> Path | None:
        if value in (None, ""):
            return None
        return _resolve_path(value, info)

    @model_validator(mode="after")
    def _validate_paths(self) -> "ReportConfig":
        if self.output_path.suffix.lower() not in {".md", ".html", ".txt"}:
            raise ValueError("output_path must end with .md, .html or .txt")
        return self
