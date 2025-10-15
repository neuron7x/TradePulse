"""Parquet lake adaptor for TradePulse time-series data."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Iterable, Sequence

from .base import TimeSeriesAdapter, TimeSeriesPoint


class ParquetTimeSeriesAdapter(TimeSeriesAdapter):
    """Persist series into partitioned Parquet datasets."""

    _MODULES: ClassVar[tuple[Any, Any, Any] | None] = None

    def __init__(self, root_path: str | Path, *, partitioning: str = "date") -> None:
        self._root = Path(root_path)
        self._partitioning = partitioning
        self._root.mkdir(parents=True, exist_ok=True)

    def _load_pyarrow(self) -> tuple[Any, Any, Any]:
        cached = ParquetTimeSeriesAdapter._MODULES
        if cached is not None:
            return cached
        spec = importlib.util.find_spec("pyarrow")
        if spec is None:
            msg = "pyarrow package is required for ParquetTimeSeriesAdapter"
            raise RuntimeError(msg)
        module = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
        dataset = importlib.import_module("pyarrow.dataset")
        ParquetTimeSeriesAdapter._MODULES = (module, parquet, dataset)
        return ParquetTimeSeriesAdapter._MODULES

    @staticmethod
    def _normalise_tags(tags: dict[str, Any] | None) -> dict[str, str]:
        if not tags:
            return {}
        return {str(key): str(value) for key, value in tags.items()}

    @staticmethod
    def _normalise_values(values: dict[str, Any]) -> dict[str, float]:
        normalised: dict[str, float] = {}
        for key, value in values.items():
            if value is None:
                continue
            normalised[str(key)] = float(value)
        return normalised

    def _resolve_path(self, timestamp: datetime, table: str) -> Path:
        if self._partitioning == "date":
            date_part = timestamp.strftime("%Y/%m/%d")
            return self._root / table / date_part
        return self._root / table

    def write_points(self, table: str, points: Sequence[TimeSeriesPoint]) -> int:
        if not points:
            return 0
        pyarrow_module, parquet_module, _ = self._load_pyarrow()
        batches = {}
        for point in points:
            path = self._resolve_path(point.timestamp, table)
            path.mkdir(parents=True, exist_ok=True)
            batches.setdefault(path, []).append(point)
        for path, bucket in batches.items():
            data = {
                "timestamp": [p.timestamp for p in bucket],
                "tags": [self._normalise_tags(p.tags) for p in bucket],
                "values": [self._normalise_values(p.values) for p in bucket],
            }
            table_obj = pyarrow_module.Table.from_pydict(data)
            file_path = path / "data.parquet"
            parquet_module.write_table(table_obj, file_path, append=file_path.exists())
        return len(points)

    def read_points(
        self,
        table: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterable[TimeSeriesPoint]:
        dataset_path = self._root / table
        if not dataset_path.exists():
            return
        pyarrow_module, _, dataset_module = self._load_pyarrow()
        try:
            dataset = dataset_module.dataset(
                dataset_path,
                format="parquet",
                partitioning="hive",
            )
        except (FileNotFoundError, OSError, ValueError):
            return
        expression = None
        if start is not None:
            expression = dataset_module.field("timestamp") >= pyarrow_module.scalar(start)
        if end is not None:
            end_expr = dataset_module.field("timestamp") <= pyarrow_module.scalar(end)
            expression = end_expr if expression is None else expression & end_expr
        table = dataset.to_table(filter=expression, limit=limit)
        for batch in table.to_pylist():
            point = TimeSeriesPoint(
                timestamp=batch["timestamp"],
                tags=self._normalise_tags(batch.get("tags")),
                values=self._normalise_values(batch.get("values") or {}),
            )
            yield point

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None
