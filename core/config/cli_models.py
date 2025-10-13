"""Lightweight Pydantic models for TradePulse CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "CatalogConfig",
    "DataSourceConfig",
    "ExecutionConfig",
    "IngestConfig",
    "MaterializeConfig",
    "OptimizeConfig",
    "ReportConfig",
    "ServeConfig",
    "StrategyConfig",
    "TrainConfig",
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


class FrameSourceConfig(BaseModel):
    """Simplified frame source for commands that only require raw tabular data."""

    kind: Literal["csv", "parquet"] = "csv"
    path: Path


class TrainDataConfig(FrameSourceConfig):
    """Dataset description for the ``train`` command."""

    signal_field: str = "signal"
    reward_field: str = "reward"
    kappa_field: str = "kappa"


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


class MaterializeConfig(TradePulseBaseConfig):
    """Configuration for streaming feature materialisation."""

    source: FrameSourceConfig
    feature_view: str
    store_root: Path = Field(default=Path("data/online"))
    checkpoint_path: Path = Field(default=Path("data/checkpoints/materialize.json"))
    microbatch_size: int = 500
    dedup_keys: List[str] = Field(default_factory=lambda: ["entity_id", "ts"])
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)

    @model_validator(mode="after")
    def _validate_materialize(self) -> "MaterializeConfig":
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if not self.dedup_keys:
            raise ValueError("dedup_keys cannot be empty")
        return self


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


class ServeConfig(ExecConfig):
    """Configuration for publishing the latest signal via ``serve``."""

    results_path: Path = Field(default=Path("reports/serve.json"))


class ReportConfig(TradePulseBaseConfig):
    """Configuration for aggregating CLI outputs into a report."""

    inputs: List[Path]
    output_path: Path
    html_output_path: Path | None = None
    pdf_output_path: Path | None = None
    template: Optional[Path] = None
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)


class CalibrationSearchConfig(BaseModel):
    """Hyperparameter search ranges for Adaptive Market Mind calibration."""

    iters: int = 200
    seed: int = 7
    ema_span: Tuple[int, int] = (8, 96)
    vol_lambda: Tuple[float, float] = (0.86, 0.98)
    alpha: Tuple[float, float] = (0.2, 5.0)
    beta: Tuple[float, float] = (0.1, 2.0)
    lambda_sync: Tuple[float, float] = (0.2, 1.2)
    eta_ricci: Tuple[float, float] = (0.1, 1.0)
    rho: Tuple[float, float] = (0.01, 0.12)

    @model_validator(mode="after")
    def _validate_ranges(self) -> "CalibrationSearchConfig":
        if self.iters < 0:
            raise ValueError("iters must be non-negative")
        return self


class TrainConfig(TradePulseBaseConfig):
    """Configuration for calibrating an Adaptive Market Mind."""

    data: TrainDataConfig
    results_path: Path = Field(default=Path("reports/train.json"))
    calibration: CalibrationSearchConfig = Field(default_factory=CalibrationSearchConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
