"""Utilities for dataframe serialization with optional parquet backends."""

from __future__ import annotations

import io
import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

__all__ = [
    "MissingParquetDependencyError",
    "dataframe_to_parquet_bytes",
    "purge_dataframe_artifacts",
    "read_dataframe",
    "write_dataframe",
    "reset_dataframe_io_backends",
]


class MissingParquetDependencyError(RuntimeError):
    """Raised when no parquet-capable backend is available."""


@dataclass(frozen=True)
class _Backend:
    name: str
    suffix: str
    write_fn: Callable[[pd.DataFrame, Path, bool], None]
    read_fn: Callable[[Path], pd.DataFrame]
    to_bytes_fn: Callable[[pd.DataFrame, bool], bytes] | None


_PARQUET_SUFFIX = ".parquet"
_JSON_SUFFIX = ".json"


@lru_cache(maxsize=1)
def _pyarrow_available() -> bool:
    try:
        importlib.import_module("pyarrow")
    except ModuleNotFoundError:
        return False
    return True


@lru_cache(maxsize=1)
def _load_polars() -> object:
    import polars as pl  # type: ignore

    return pl


def reset_dataframe_io_backends() -> None:
    """Reset cached backend discovery (useful for tests)."""

    _pyarrow_available.cache_clear()
    _load_polars.cache_clear()


def _pyarrow_backend() -> _Backend:
    def _write(frame: pd.DataFrame, path: Path, index: bool) -> None:
        frame.to_parquet(path, engine="pyarrow", index=index)

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path, engine="pyarrow")

    def _to_bytes(frame: pd.DataFrame, index: bool) -> bytes:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, engine="pyarrow", index=index)
        return buffer.getvalue()

    return _Backend("pyarrow", _PARQUET_SUFFIX, _write, _read, _to_bytes)


def _polars_backend() -> _Backend:
    try:
        pl = _load_polars()
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise MissingParquetDependencyError("polars is not installed") from exc

    def _prepare(frame: pd.DataFrame, index: bool):
        if index:
            payload = frame.reset_index()
        else:
            payload = frame.reset_index(drop=True)
        return pl.from_pandas(payload)

    def _write(frame: pd.DataFrame, path: Path, index: bool) -> None:
        buffer = io.BytesIO()
        _prepare(frame, index).write_parquet(buffer)
        path.write_bytes(buffer.getvalue())

    def _read(path: Path) -> pd.DataFrame:
        dataset = pl.read_parquet(path)
        return dataset.to_pandas(use_pyarrow=False)

    def _to_bytes(frame: pd.DataFrame, index: bool) -> bytes:
        buffer = io.BytesIO()
        _prepare(frame, index).write_parquet(buffer)
        return buffer.getvalue()

    return _Backend("polars", _PARQUET_SUFFIX, _write, _read, _to_bytes)


def _json_backend() -> _Backend:
    def _write(frame: pd.DataFrame, path: Path, index: bool) -> None:
        frame.to_json(
            path,
            orient="split",
            index=index,
            date_format="iso",
            date_unit="ns",
            double_precision=15,
        )

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_json(path, orient="split")

    return _Backend("json", _JSON_SUFFIX, _write, _read, None)


def _available_backends() -> list[_Backend]:
    backends: list[_Backend] = []
    if _pyarrow_available():
        backends.append(_pyarrow_backend())
    try:
        backends.append(_polars_backend())
    except MissingParquetDependencyError:
        pass
    backends.append(_json_backend())
    return backends


def _select_backend(require_parquet: bool, allow_json_fallback: bool) -> _Backend:
    for backend in _available_backends():
        if backend.suffix == _PARQUET_SUFFIX:
            return backend
        if allow_json_fallback and backend.suffix == _JSON_SUFFIX and not require_parquet:
            return backend
    raise MissingParquetDependencyError(
        "No parquet backend available. Install 'tradepulse[feature_store]' for pyarrow support or install polars."
    )


def _normalize_destination(destination: Path) -> tuple[Path, bool]:
    if destination.suffix:
        base = destination.with_suffix("")
        explicit_suffix = True
    else:
        base = destination
        explicit_suffix = False
    return base, explicit_suffix


def write_dataframe(
    frame: pd.DataFrame,
    destination: Path,
    *,
    index: bool = False,
    allow_json_fallback: bool = False,
) -> Path:
    """Serialize ``frame`` to ``destination`` using the first available backend."""

    base, explicit_suffix = _normalize_destination(destination)
    require_parquet = destination.suffix.lower() == _PARQUET_SUFFIX if explicit_suffix else False
    backend = _select_backend(require_parquet, allow_json_fallback)
    target = base.with_suffix(backend.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    backend.write_fn(frame, target, index)
    return target


def read_dataframe(path: Path, *, allow_json_fallback: bool = False) -> pd.DataFrame:
    """Load a dataframe from ``path`` using the first compatible backend."""

    path = Path(path)
    if path.suffix:
        suffix = path.suffix.lower()
        if suffix == _PARQUET_SUFFIX:
            if _pyarrow_available():
                return _pyarrow_backend().read_fn(path)
            try:
                return _polars_backend().read_fn(path)
            except MissingParquetDependencyError as exc:
                raise MissingParquetDependencyError(
                    "Unable to read parquet file without pyarrow or polars. Install 'tradepulse[feature_store]'."
                ) from exc
        if allow_json_fallback and suffix == _JSON_SUFFIX:
            return _json_backend().read_fn(path)
        raise ValueError(f"Unsupported dataframe suffix '{suffix}'")

    base = path
    parquet_path = base.with_suffix(_PARQUET_SUFFIX)
    if parquet_path.exists():
        return read_dataframe(parquet_path, allow_json_fallback=allow_json_fallback)
    json_path = base.with_suffix(_JSON_SUFFIX)
    if allow_json_fallback and json_path.exists():
        return _json_backend().read_fn(json_path)
    return pd.DataFrame()


def purge_dataframe_artifacts(base_path: Path) -> None:
    """Remove serialized dataframe artefacts for ``base_path``."""

    for suffix in (_PARQUET_SUFFIX, _JSON_SUFFIX):
        candidate = base_path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def dataframe_to_parquet_bytes(frame: pd.DataFrame, *, index: bool = False) -> bytes:
    """Return a parquet-encoded payload for ``frame`` using the preferred backend."""

    backend = _select_backend(require_parquet=True, allow_json_fallback=False)
    if backend.to_bytes_fn is None:
        raise MissingParquetDependencyError(
            "The selected backend does not support parquet serialization to bytes."
        )
    return backend.to_bytes_fn(frame, index)
