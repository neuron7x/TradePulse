# SPDX-License-Identifier: MIT
"""Pytest fixtures and environment setup.

This module performs two responsibilities:

* Ensure the repository root is importable so tests can resolve in-tree
  packages without installing them.
* Provide graceful fallbacks for optional plugins (e.g. ``pytest-cov``)
  that may be absent in constrained environments.  The CI workflow runs
  ``pytest`` with ``--cov``/``--cov-report`` switches; without the
  ``pytest-cov`` plugin, pytest would reject those flags.  We register
  lightweight no-op handlers so the options are accepted while keeping
  behaviour identical when ``pytest-cov`` is available.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
import types
import warnings
from typing import Iterable

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "exchange_calendars" not in sys.modules:  # pragma: no cover - optional dependency shim
    stub = types.ModuleType("exchange_calendars")

    class _AlwaysOpenCalendar:
        tz = types.SimpleNamespace(key="UTC")

        def valid_days(self, start, end):  # noqa: D401 - mimic API
            """Return continuous range between ``start`` and ``end``."""

            return []

    class _AlwaysOpenNamespace:
        @staticmethod
        def AlwaysOpenCalendar():  # noqa: N802 - mirrors third-party API
            return _AlwaysOpenCalendar()

    class _ErrorsNamespace:
        class InvalidCalendarName(Exception):
            pass

    def _get_calendar(name: str):  # noqa: D401 - mimic API signature
        return _AlwaysOpenCalendar()

    stub.always_open = _AlwaysOpenNamespace()
    stub.errors = _ErrorsNamespace()
    stub.ExchangeCalendar = object
    stub.get_calendar = _get_calendar
    stub.resolve_alias = lambda value: value  # type: ignore[assignment]
    sys.modules["exchange_calendars"] = stub


def _register_noop_cov_options(parser: "pytest.Parser") -> None:
    """Register ``--cov`` flags when ``pytest-cov`` is unavailable.

    When the real plugin is present it will have already added the
    options, in which case re-registering raises ``ValueError``—we
    silently ignore that scenario so the genuine implementation wins.
    """

    group = parser.getgroup("cov", "coverage reporting")
    options: Iterable[tuple[str, dict[str, object]]] = (
        ("--cov", {"action": "append", "dest": "tradepulse_cov", "metavar": "PATH", "default": []}),
        (
            "--cov-report",
            {
                "action": "append",
                "dest": "tradepulse_cov_report",
                "metavar": "TYPE",
                "default": [],
            },
        ),
    )
    for opt, kwargs in options:
        try:
            group.addoption(opt, **kwargs)
        except ValueError:
            # Option already registered (e.g. by pytest-cov); respect the original.
            pass


def pytest_addoption(parser):  # type: ignore[override]
    try:
        import pytest_cov.plugin  # noqa: F401  # type: ignore[attr-defined]
    except Exception:
        _register_noop_cov_options(parser)


def pytest_configure(config):  # type: ignore[override]
    if config.pluginmanager.hasplugin("pytest_cov"):
        return
    cov_targets = config.getoption("tradepulse_cov", default=None)
    cov_reports = config.getoption("tradepulse_cov_report", default=None)
    if cov_targets or cov_reports:
        from _pytest.warning_types import PytestWarning

        warnings.warn(
            "pytest-cov is not installed; coverage options are accepted but ignored.",
            PytestWarning,
            stacklevel=2,
        )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):  # type: ignore[override]
    """Provide a minimal ``pytest-asyncio`` fallback."""

    if pyfuncitem.config.pluginmanager.hasplugin("pytest_asyncio"):
        return None

    testfunction = pyfuncitem.obj
    if not asyncio.iscoroutinefunction(testfunction):
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        kwargs = {arg: pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
        loop.run_until_complete(testfunction(**kwargs))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    return True
