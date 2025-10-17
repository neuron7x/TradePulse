"""Context-managed SQLAlchemy session orchestration with read/write routing."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import Lock

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .engine import warm_pool

__all__ = ["SessionManager"]


class SessionManager:
    """Route ORM sessions to writer or reader pools transparently."""

    def __init__(
        self,
        writer_engine: Engine,
        reader_engines: Sequence[Engine] | None = None,
        *,
        expire_on_commit: bool = False,
        owns_engines: bool = True,
    ) -> None:
        self._writer_engine = writer_engine
        self._reader_engines = tuple(reader_engines or ())
        self._writer_factory = sessionmaker(
            bind=self._writer_engine,
            autoflush=False,
            expire_on_commit=expire_on_commit,
            future=True,
        )
        self._reader_factories = tuple(
            sessionmaker(
                bind=engine,
                autoflush=False,
                expire_on_commit=expire_on_commit,
                future=True,
            )
            for engine in self._reader_engines
        )
        self._reader_cycle = itertools.cycle(self._reader_factories) if self._reader_factories else None
        self._lock = Lock()
        self._owns_engines = owns_engines
        self._closed = False

    @contextmanager
    def session(self, *, read_only: bool = False) -> Iterator[Session]:
        """Yield a session bound to either the writer or a reader replica."""

        factory = self._select_factory(read_only=read_only)
        session: Session = factory()
        try:
            yield session
            if read_only:
                session.rollback()
            else:
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def warmup(self, *, writer_connections: int = 0, reader_connections: int = 0) -> None:
        """Pre-open connections so latency-sensitive workloads do not pay the initial cost."""

        warm_pool(self._writer_engine, target_size=writer_connections)
        if reader_connections <= 0:
            reader_target = 0
        else:
            reader_target = reader_connections
        for engine in self._reader_engines:
            warm_pool(engine, target_size=reader_target)

    def close(self) -> None:
        """Dispose all underlying SQLAlchemy engines if the manager owns them."""

        if not self._owns_engines:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._writer_engine.dispose()
        for engine in self._reader_engines:
            engine.dispose()

    @property
    def writer_engine(self) -> Engine:
        return self._writer_engine

    @property
    def reader_engines(self) -> tuple[Engine, ...]:
        return self._reader_engines

    def _select_factory(self, *, read_only: bool) -> sessionmaker[Session]:
        if read_only and self._reader_cycle is not None:
            with self._lock:
                return next(self._reader_cycle)
        return self._writer_factory
