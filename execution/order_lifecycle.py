"""Persistence helpers for order lifecycle transitions.

This module centralises the SQL used to persist and retrieve order lifecycle
information.  Bandit previously reported potential SQL injection vectors
because queries were assembled via string interpolation.  To mitigate this we
now rely on :mod:`psycopg.sql` composables (or a minimal local fallback when
``psycopg`` is unavailable) so table identifiers are quoted correctly and query
parts are never concatenated with untrusted input.

The resulting helpers expose small, well-typed building blocks that other
components can use alongside :class:`libs.db.access.DataAccessLayer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Sequence

try:  # pragma: no cover - optional dependency boundary
    from psycopg import sql
except Exception:  # pragma: no cover - fallback used when psycopg is absent

    class _Composable:
        """Lightweight stand-in for :class:`psycopg.sql.Composable`."""

        __slots__ = ("_value",)

        def __init__(self, value: str) -> None:
            self._value = value

        def format(self, **mapping: "_Composable | str") -> "_Composable":
            rendered = self._value
            for key, replacement in mapping.items():
                if isinstance(replacement, _Composable):
                    token = str(replacement)
                else:
                    token = str(replacement)
                rendered = rendered.replace("{" + key + "}", token)
            return _Composable(rendered)

        def join(self, parts: Iterable["_Composable | str"]) -> "_Composable":
            rendered = str(self).join(str(part) for part in parts)
            return _Composable(rendered)

        def __str__(self) -> str:  # pragma: no cover - trivial accessor
            return self._value

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            return f"_Composable({self._value!r})"

        def __add__(self, other: object) -> "_Composable":  # pragma: no cover
            return _Composable(self._value + str(other))

    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def _quote_identifier(value: str) -> str:
        if not _IDENTIFIER_RE.match(value):
            raise ValueError(f"invalid SQL identifier: {value!r}")
        return '"' + value.replace('"', '""') + '"'

    class _Identifier(_Composable):
        def __init__(self, *parts: str) -> None:
            quoted = ".".join(_quote_identifier(part) for part in parts if part)
            super().__init__(quoted)

    class _Placeholder(_Composable):
        def __init__(self, value: str) -> None:
            super().__init__(value)

    class _SQLModule:  # pragma: no cover - trivial adapter
        SQL = staticmethod(lambda value: _Composable(str(value)))
        Identifier = staticmethod(lambda *parts: _Identifier(*parts))
        Placeholder = staticmethod(lambda name=None: _Placeholder("%s" if name is None else f"%({name})s"))

    sql = _SQLModule()  # type: ignore[assignment]


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Ensure identifiers follow the project's defensive SQL policy."""

    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{field_name} must be a valid SQL identifier, got {value!r}")
    return value


@dataclass(slots=True)
class _PlaceholderFactory:
    """Generate driver-compatible placeholders for a given paramstyle."""

    paramstyle: str = "pyformat"
    _index: int = field(default=0, init=False, repr=False)

    def __call__(self, name: str | None = None) -> object:
        style = self.paramstyle
        if style == "qmark":
            return sql.SQL("?")
        if style == "numeric":
            self._index += 1
            return sql.SQL(f":{self._index}")
        if style == "named":
            if not name:
                raise ValueError("named paramstyle requires parameter names")
            return sql.SQL(f":{name}")
        if style == "format":
            return sql.SQL("%s")
        if style == "pyformat":
            if name:
                return sql.SQL(f"%({name})s")
            return sql.SQL("%s")
        raise ValueError(f"unsupported paramstyle: {style}")


@dataclass(slots=True)
class OrderLifecycleQueries:
    """Produce parametrised SQL statements for lifecycle operations."""

    table: str = "order_lifecycle_events"
    schema: str | None = "execution"
    paramstyle: str = "pyformat"

    def __post_init__(self) -> None:
        self.table = _validate_identifier(self.table, field_name="table")
        if self.schema is not None:
            self.schema = _validate_identifier(self.schema, field_name="schema")
        identifiers = (self.schema, self.table) if self.schema else (self.table,)
        self._qualified_table = sql.Identifier(*[part for part in identifiers if part])
        self._columns = (
            "order_id",
            "correlation_id",
            "event",
            "from_status",
            "to_status",
            "details",
        )
        self._column_list = sql.SQL(", ").join(sql.Identifier(col) for col in self._columns)

    # ------------------------------------------------------------------
    def insert_event(self) -> object:
        """Return a parametrised INSERT statement."""

        factory = _PlaceholderFactory(self.paramstyle)
        placeholders = [factory(name) for name in self._columns]
        values = sql.SQL(", ").join(placeholders)
        return sql.SQL(
            "INSERT INTO {table} ({columns}) VALUES ({values}) "
            "ON CONFLICT (order_id, correlation_id) DO NOTHING"
        ).format(table=self._qualified_table, columns=self._column_list, values=values)

    def select_by_order_and_correlation(self) -> object:
        """Return a SELECT filtered by order and correlation identifiers."""

        factory = _PlaceholderFactory(self.paramstyle)
        order_placeholder, correlation_placeholder = (
            factory("order_id"),
            factory("correlation_id"),
        )
        return sql.SQL(
            "SELECT sequence, order_id, correlation_id, event, from_status, "
            "to_status, details, created_at "
            "FROM {table} "
            "WHERE order_id = {order_id} AND correlation_id = {correlation_id}"
        ).format(
            table=self._qualified_table,
            order_id=order_placeholder,
            correlation_id=correlation_placeholder,
        )

    def select_by_order(self) -> object:
        """Return a SELECT statement retrieving the full lifecycle for an order."""

        factory = _PlaceholderFactory(self.paramstyle)
        order_placeholder = factory("order_id")
        return sql.SQL(
            "SELECT sequence, order_id, correlation_id, event, from_status, "
            "to_status, details, created_at "
            "FROM {table} WHERE order_id = {order_id} ORDER BY sequence"
        ).format(table=self._qualified_table, order_id=order_placeholder)

    def select_latest_per_order(self) -> object:
        """Return a SELECT producing the latest lifecycle event per order."""

        return sql.SQL(
            "SELECT t.sequence, t.order_id, t.correlation_id, t.event, "
            "t.from_status, t.to_status, t.details, t.created_at "
            "FROM {table} AS t "
            "JOIN ("
            "    SELECT order_id, MAX(sequence) AS max_sequence "
            "    FROM {table} GROUP BY order_id"
            ") AS latest "
            "ON t.order_id = latest.order_id AND t.sequence = latest.max_sequence"
        ).format(table=self._qualified_table)

    def select_most_recent_for_order(self) -> object:
        """Return a SELECT retrieving the most recent event for an order."""

        factory = _PlaceholderFactory(self.paramstyle)
        order_placeholder = factory("order_id")
        return sql.SQL(
            "SELECT sequence, order_id, correlation_id, event, from_status, "
            "to_status, details, created_at "
            "FROM {table} WHERE order_id = {order_id} "
            "ORDER BY sequence DESC LIMIT 1"
        ).format(table=self._qualified_table, order_id=order_placeholder)

    # ------------------------------------------------------------------
    def history_for_orders(self, order_ids: Sequence[str]) -> object:
        """Return a SELECT fetching events for the supplied order identifiers."""

        if not order_ids:
            raise ValueError("order_ids cannot be empty")

        factory = _PlaceholderFactory(self.paramstyle)
        placeholders = [factory(f"order_id_{idx}") for idx, _ in enumerate(order_ids, start=1)]
        placeholder_list = sql.SQL(", ").join(placeholders)
        return sql.SQL(
            "SELECT sequence, order_id, correlation_id, event, from_status, "
            "to_status, details, created_at "
            "FROM {table} WHERE order_id IN ({order_ids}) ORDER BY order_id, sequence"
        ).format(table=self._qualified_table, order_ids=placeholder_list)


__all__ = ["OrderLifecycleQueries"]
