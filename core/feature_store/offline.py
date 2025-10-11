"""Offline feature store implementations."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency import
    import pyarrow as _pa
    import pyarrow.dataset as _ds
    import pyarrow.parquet as _pq
except ImportError:  # pragma: no cover - optional dependency import
    _pa = _ds = _pq = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - used only for static type checking
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

from .config import OfflineStoreConfig
from .models import FeatureSet

WriteMode = Literal["append", "overwrite"]


class OfflineStore(ABC):
    """Common interface implemented by all offline stores."""

    def __init__(self, config: OfflineStoreConfig) -> None:
        self.config = config

    @abstractmethod
    def write(self, feature_set: FeatureSet, mode: WriteMode = "append") -> Path:
        """Persist the feature set and return the materialized location."""

    @abstractmethod
    def read(
        self,
        feature_set_name: str,
        columns: Optional[Sequence[str]] = None,
        entity_filter: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Load features for the requested feature set."""

    @abstractmethod
    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        """Return the max timestamp stored for the feature set."""

    def _ensure_partition_columns(self, feature_set: FeatureSet) -> None:
        missing = [
            column for column in self.config.partition_by if column not in feature_set.dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"Partition columns {missing} declared in configuration are missing from '{feature_set.name}'"
            )


class ParquetOfflineStore(OfflineStore):
    """Offline store backed by partitioned Parquet datasets."""

    def __init__(self, config: OfflineStoreConfig) -> None:
        if config.format != "parquet":
            raise ValueError("ParquetOfflineStore must be configured with format='parquet'")
        super().__init__(config)

    def _dataset_path(self, feature_set_name: str) -> Path:
        return self.config.base_path / feature_set_name

    def _require_pyarrow(self) -> Tuple[Any, Any, Any]:
        if _pa is None or _ds is None or _pq is None:
            raise RuntimeError(
                "Parquet support requires the optional dependency 'pyarrow'. "
                "Install TradePulse with `pip install tradepulse[feature-store]` to enable it."
            )
        return _pa, _ds, _pq

    def write(self, feature_set: FeatureSet, mode: WriteMode = "append") -> Path:
        self._ensure_partition_columns(feature_set)
        dataset_path = self._dataset_path(feature_set.name)
        if mode == "overwrite" and dataset_path.exists():
            shutil.rmtree(dataset_path)
        pa, _, pq = self._require_pyarrow()
        table = pa.Table.from_pandas(feature_set.dataframe, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=dataset_path,
            partition_cols=self.config.partition_by or None,
            compression=self.config.compression,
            basename_template="part-{i}.parquet",
        )
        return dataset_path

    def read(
        self,
        feature_set_name: str,
        columns: Optional[Sequence[str]] = None,
        entity_filter: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        _, ds, _ = self._require_pyarrow()
        dataset_path = self._dataset_path(feature_set_name)
        if not dataset_path.exists():
            return pd.DataFrame()
        dataset = ds.dataset(dataset_path, format="parquet")
        arrow_table = dataset.to_table(columns=list(columns) if columns else None)
        frame = arrow_table.to_pandas()
        if entity_filter is not None and not entity_filter.empty:
            key_columns = list(entity_filter.columns)
            frame = frame.merge(entity_filter.drop_duplicates(), on=key_columns, how="inner")
        return frame

    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        dataset_path = self._dataset_path(feature_set_name)
        if not dataset_path.exists():
            return None
        _, ds, _ = self._require_pyarrow()
        dataset = ds.dataset(dataset_path, format="parquet")
        arrow_table = dataset.to_table(columns=[timestamp_column])
        if arrow_table.num_rows == 0:
            return None
        return arrow_table.to_pandas()[timestamp_column].max()


class DeltaOfflineStore(OfflineStore):
    """Offline store backed by Delta Lake tables."""

    def __init__(self, config: OfflineStoreConfig) -> None:
        try:
            from deltalake import DeltaTable, write_deltalake
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Delta Lake support requires the 'deltalake' extra (pip install tradepulse[feature-store])"
            ) from exc
        super().__init__(config)
        self._delta_table_cls = DeltaTable
        self._write_deltalake = write_deltalake

    def _table_uri(self, feature_set_name: str) -> str:
        return str(self.config.base_path / feature_set_name)

    def write(self, feature_set: FeatureSet, mode: WriteMode = "append") -> Path:
        delta_mode = "append" if mode == "append" else "overwrite"
        self._write_deltalake(
            self._table_uri(feature_set.name),
            feature_set.dataframe,
            mode=delta_mode,
            overwrite_schema=True,
        )
        return Path(self._table_uri(feature_set.name))

    def _load_table(self, feature_set_name: str):
        table_uri = self._table_uri(feature_set_name)
        if not Path(table_uri).exists():
            return None
        return self._delta_table_cls(table_uri)

    def read(
        self,
        feature_set_name: str,
        columns: Optional[Sequence[str]] = None,
        entity_filter: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        table = self._load_table(feature_set_name)
        if table is None:
            return pd.DataFrame()
        arrow_table = table.to_pyarrow_table(columns=list(columns) if columns else None)
        frame = arrow_table.to_pandas()
        if entity_filter is not None and not entity_filter.empty:
            frame = frame.merge(entity_filter.drop_duplicates(), on=list(entity_filter.columns), how="inner")
        return frame

    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        table = self._load_table(feature_set_name)
        if table is None:
            return None
        arrow_table = table.to_pyarrow_table(columns=[timestamp_column])
        if arrow_table.num_rows == 0:
            return None
        return arrow_table.to_pandas()[timestamp_column].max()


class IcebergOfflineStore(ParquetOfflineStore):
    """Offline store targeting Apache Iceberg tables.

    The implementation relies on PyIceberg for schema compatibility but keeps the
    underlying storage in Parquet format, providing a portable fallback when a
    catalog is not available during local development.
    """

    def __init__(self, config: OfflineStoreConfig) -> None:
        try:
            import pyiceberg  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Iceberg support requires the 'pyiceberg' extra (pip install tradepulse[feature-store])"
            ) from exc
        super().__init__(OfflineStoreConfig(**{**config.model_dump(), "format": "parquet"}))


class OfflineStoreFactory:
    """Factory producing offline store implementations based on configuration."""

    def __init__(self, config: OfflineStoreConfig) -> None:
        self.config = config

    def create(self) -> OfflineStore:
        if self.config.format == "parquet":
            return ParquetOfflineStore(self.config)
        if self.config.format == "delta":
            return DeltaOfflineStore(self.config)
        if self.config.format == "iceberg":
            return IcebergOfflineStore(self.config)
        raise ValueError(f"Unsupported offline store format: {self.config.format}")


__all__ = [
    "DeltaOfflineStore",
    "IcebergOfflineStore",
    "OfflineStore",
    "OfflineStoreFactory",
    "ParquetOfflineStore",
    "WriteMode",
]
