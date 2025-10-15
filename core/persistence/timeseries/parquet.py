"""Parquet lake adaptor for TradePulse time-series data."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .base import TimeSeriesAdapter, TimeSeriesPoint


class ParquetTimeSeriesAdapter(TimeSeriesAdapter):
    """Persist series into partitioned Parquet datasets."""

    def __init__(self, root_path: str | Path, *, partitioning: str = "date") -> None:
        self._root = Path(root_path)
        self._partitioning = partitioning
        self._pyarrow = self._load_pyarrow()
        self._root.mkdir(parents=True, exist_ok=True)

    def _load_pyarrow(self):
        spec = importlib.util.find_spec("pyarrow")
        if spec is None:
            msg = "pyarrow package is required for ParquetTimeSeriesAdapter"
            raise RuntimeError(msg)
        module = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
        dataset = importlib.import_module("pyarrow.dataset")
        return module, parquet, dataset

    def _resolve_path(self, timestamp: datetime, table: str) -> Path:
        if self._partitioning == "date":
            date_part = timestamp.strftime("%Y/%m/%d")
            return self._root / table / date_part
        return self._root / table

    def write_points(self, table: str, points: Sequence[TimeSeriesPoint]) -> int:
        if not points:
            return 0
        pyarrow, parquet_mod, dataset_mod = self._pyarrow
        batches = {}
        for point in points:
            path = self._resolve_path(point.timestamp, table)
            path.mkdir(parents=True, exist_ok=True)
            batches.setdefault(path, []).append(point)
        for path, bucket in batches.items():
            data = {
                "timestamp": [p.timestamp for p in bucket],
                "tags": [p.tags or {} for p in bucket],
                "values": [p.values for p in bucket],
            }
            table_obj = pyarrow.Table.from_pydict(data)
            file_path = path / "data.parquet"
            if file_path.exists():
                existing = parquet_mod.read_table(file_path)
                combined = pyarrow.concat_tables([existing, table_obj])
                parquet_mod.write_table(combined, file_path)
            else:
                parquet_mod.write_table(table_obj, file_path)
        return len(points)

    def read_points(
        self,
        table: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> Iterable[TimeSeriesPoint]:
        pyarrow, parquet_mod, dataset_mod = self._pyarrow
        dataset_path = self._root / table
        if not dataset_path.exists():
            return iter(())
        dataset = dataset_mod.dataset(dataset_path, format="parquet", partitioning="hive")
        expression = None
        if start is not None:
            expression = dataset_mod.field("timestamp") >= pyarrow.scalar(start)
        if end is not None:
            end_expr = dataset_mod.field("timestamp") <= pyarrow.scalar(end)
            expression = end_expr if expression is None else expression & end_expr
        table = dataset.to_table(filter=expression, limit=limit)
        count = 0
        for batch in table.to_pylist():
            point = TimeSeriesPoint(
                timestamp=batch["timestamp"],
                tags=batch.get("tags") or {},
                values=batch.get("values") or {},
            )
            yield point
            count += 1

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None
