"""Configuration models for the feature store components."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OfflineStoreConfig(BaseModel):
    """Configuration for offline feature persistence."""

    format: Literal["parquet", "delta", "iceberg"] = "parquet"
    base_path: Path = Field(..., description="Filesystem location for offline feature sets.")
    compression: str = Field("snappy", description="Compression codec to use when persisting data.")
    partition_by: List[str] = Field(default_factory=list, description="Columns used for partitioning datasets.")

    @model_validator(mode="after")
    def _ensure_directory(self) -> "OfflineStoreConfig":
        self.base_path.mkdir(parents=True, exist_ok=True)
        return self


class OnlineStoreConfig(BaseModel):
    """Configuration for online feature serving stores."""

    backend: Literal["redis", "sqlite"]
    sqlite_path: Optional[Path] = Field(None, description="Path to the SQLite database used for online serving.")
    redis_host: str = Field("localhost", description="Redis host name.")
    redis_port: int = Field(6379, description="Redis port.")
    redis_db: int = Field(0, description="Redis database index.")
    redis_username: Optional[str] = Field(None, description="Optional Redis username.")
    redis_password: Optional[str] = Field(None, description="Optional Redis password.")
    redis_ssl: bool = Field(False, description="Whether to use TLS when connecting to Redis.")
    redis_ttl_seconds: Optional[int] = Field(
        None,
        description="Optional TTL in seconds applied to materialized feature rows in Redis.",
    )

    @model_validator(mode="after")
    def _validate_backend(self) -> "OnlineStoreConfig":
        if self.backend == "sqlite" and self.sqlite_path is None:
            raise ValueError("sqlite_path must be provided when backend is 'sqlite'")
        if self.backend == "redis" and self.sqlite_path is not None:
            raise ValueError("sqlite_path should not be provided when backend is 'redis'")
        return self

    @field_validator("redis_ttl_seconds")
    @classmethod
    def _ttl_positive(cls, ttl: Optional[int]) -> Optional[int]:
        if ttl is not None and ttl <= 0:
            raise ValueError("redis_ttl_seconds must be positive when provided")
        return ttl


class MaterializationConfig(BaseModel):
    """Configuration shared by materialization flows."""

    entity_columns: List[str]
    timestamp_column: str = Field("event_timestamp", description="Column containing feature event timestamps.")
    allow_overwrite: bool = Field(
        False,
        description="Whether online materialization is allowed to overwrite fresher data.",
    )


class FeatureStoreConfig(BaseModel):
    """Root configuration combining offline and online stores."""

    offline: OfflineStoreConfig
    online: OnlineStoreConfig
    materialization: MaterializationConfig

    def entity_columns(self) -> List[str]:
        return self.materialization.entity_columns

    def timestamp_column(self) -> str:
        return self.materialization.timestamp_column


__all__ = [
    "FeatureStoreConfig",
    "MaterializationConfig",
    "OfflineStoreConfig",
    "OnlineStoreConfig",
]
