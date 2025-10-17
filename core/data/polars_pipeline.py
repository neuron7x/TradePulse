"""Utilities for constructing zero-copy Polars pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils.logging import get_logger

try:  # pragma: no cover - optional dependency
    import polars as pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pl = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import pyarrow as pa  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pa = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from polars import DataFrame, LazyFrame  # type: ignore
    from pyarrow import MemoryPool  # type: ignore


_logger = get_logger(__name__)


def _require_polars() -> Any:
    if pl is None:  # pragma: no cover - safety net when polars absent
        raise RuntimeError(
            "polars is not installed. Install it with 'pip install polars' to use the "
            "lazy pipeline helpers."
        )
    return pl


def scan_lazy(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    memory_map: bool = True,
    low_memory: bool = True,
    row_count_name: str | None = None,
) -> "LazyFrame":
    """Create a lazy Polars scan with streaming-friendly defaults."""
    module = _require_polars()
    _logger.debug(
        "Creating Polars lazy scan",
        path=str(path),
        columns=columns,
        memory_map=memory_map,
    )
    lazy_frame = module.scan_csv(
        path,
        columns=columns,
        low_memory=low_memory,
        memory_map=memory_map,
    )
    if row_count_name is not None:
        lazy_frame = lazy_frame.with_row_count(name=row_count_name)
    return lazy_frame


def collect_streaming(
    lazy_frame: "LazyFrame",
    *,
    streaming: bool = True,
    sink: Callable[["DataFrame"], Any] | None = None,
) -> "DataFrame":
    """Collect a lazy frame with streaming enabled by default."""
    _require_polars()
    _logger.debug("Collecting Polars lazy frame", streaming=streaming)
    result = lazy_frame.collect(streaming=streaming)
    if sink is not None:
        sink(result)
    return result


def lazy_column_zero_copy(
    lazy_frame: "LazyFrame",
    column: str,
    *,
    streaming: bool = True,
) -> np.ndarray:
    """Extract a column as a zero-copy NumPy array."""
    module = _require_polars()
    if pa is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("pyarrow is required for zero-copy column extraction")
    _logger.debug("Extracting zero-copy column", column=column, streaming=streaming)
    arrow_table = (
        lazy_frame.select(module.col(column)).collect(streaming=streaming).to_arrow()
    )
    if arrow_table.num_columns != 1:
        raise ValueError("Expected a single-column selection for zero-copy extraction")
    return arrow_table.column(0).to_numpy(zero_copy_only=True)


@contextmanager
def use_arrow_memory_pool(pool: "MemoryPool"):
    """Temporarily route Arrow allocations through ``pool``."""
    if pa is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("pyarrow is required to manage the memory pool")
    try:
        previous_pool = pa.default_memory_pool()
        pa.set_memory_pool(pool)
    except AttributeError:  # pragma: no cover - pyarrow without setter
        _logger.warning("pyarrow does not support swapping the global memory pool")
        yield
    else:
        try:
            yield
        finally:
            pa.set_memory_pool(previous_pool)


def enable_global_string_cache(enable: bool = True) -> None:
    """Enable or disable the Polars global string cache."""
    module = _require_polars()
    module.Config.set_global_string_cache(enable)
