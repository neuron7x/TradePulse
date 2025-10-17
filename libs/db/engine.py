"""Helpers for building hardened SQLAlchemy engines with pooling enabled."""

from __future__ import annotations

from contextlib import suppress
from typing import Mapping

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from core.config.cli_models import PostgresTLSConfig

from .config import DatabasePoolConfig, DatabaseRuntimeConfig

__all__ = ["create_engine_from_config", "warm_pool"]


def _build_connect_args(
    tls: PostgresTLSConfig | None,
    runtime: DatabaseRuntimeConfig,
) -> dict[str, object]:
    """Return connection keyword arguments compatible with psycopg."""

    options = [f"-c statement_timeout={int(runtime.statement_timeout_ms)}", "-c timezone=UTC"]
    connect_args: dict[str, object] = {
        "connect_timeout": float(runtime.connect_timeout_seconds),
        "application_name": runtime.application_name,
        "options": " ".join(options),
    }

    if runtime.target_session_attrs is not None:
        connect_args["target_session_attrs"] = runtime.target_session_attrs

    if tls is not None:
        connect_args.update(
            {
                "sslrootcert": str(tls.ca_file),
                "sslcert": str(tls.cert_file),
                "sslkey": str(tls.key_file),
            }
        )

    return connect_args


def create_engine_from_config(
    dsn: str,
    *,
    tls: PostgresTLSConfig | None,
    pool: DatabasePoolConfig,
    runtime: DatabaseRuntimeConfig,
    echo: bool = False,
    execution_options: Mapping[str, object] | None = None,
) -> Engine:
    """Instantiate a SQLAlchemy engine backed by :class:`QueuePool`."""

    engine = create_engine(
        dsn,
        echo=echo,
        future=True,
        poolclass=QueuePool,
        pool_size=int(pool.size),
        max_overflow=int(pool.max_overflow),
        pool_timeout=None if pool.timeout is None else float(pool.timeout),
        pool_recycle=float(pool.recycle),
        pool_use_lifo=bool(pool.use_lifo),
        pool_pre_ping=True,
        connect_args=_build_connect_args(tls, runtime),
    )

    effective_execution_options = {"stream_results": True}
    if execution_options:
        effective_execution_options.update(execution_options)
    engine = engine.execution_options(**effective_execution_options)
    return engine


def warm_pool(engine: Engine, *, target_size: int) -> None:
    """Eagerly open ``target_size`` connections to prime the pool."""

    if target_size <= 0:
        return

    opened: list[object] = []
    try:
        for _ in range(target_size):
            opened.append(engine.connect())
    finally:
        for connection in opened:
            with suppress(Exception):
                connection.close()
