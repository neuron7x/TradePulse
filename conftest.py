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
import inspect
import pathlib
import sys
import warnings
from typing import Any, Iterable

import pytest

try:
    import pytest_asyncio  # type: ignore[attr-defined]  # noqa: F401
except Exception:  # pragma: no cover - import guard only
    HAS_PYTEST_ASYNCIO = False
else:  # pragma: no cover - exercised when plugin present
    HAS_PYTEST_ASYNCIO = True

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.pytest_asyncio_fallback import run_coroutine


if not HAS_PYTEST_ASYNCIO:

    @pytest.fixture
    def event_loop():  # type: ignore[override]
        """Provide a close analogue to ``pytest-asyncio``'s ``event_loop`` fixture."""

        loop = asyncio.new_event_loop()
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        asyncio.set_event_loop(loop)

        try:
            yield loop
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                asyncio.set_event_loop(previous_loop)

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
    if not config.pluginmanager.hasplugin("pytest_cov"):
        cov_targets = config.getoption("tradepulse_cov", default=None)
        cov_reports = config.getoption("tradepulse_cov_report", default=None)
        if cov_targets or cov_reports:
            from _pytest.warning_types import PytestWarning

            warnings.warn(
                "pytest-cov is not installed; coverage options are accepted but ignored.",
                PytestWarning,
                stacklevel=2,
            )

    if not config.pluginmanager.hasplugin("pytest_asyncio"):
        config.addinivalue_line(
            "markers",
            "asyncio: mark a test that relies on the asyncio event loop fallback",
        )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):  # type: ignore[override]
    """Provide a minimal ``pytest-asyncio`` fallback."""

    if pyfuncitem.config.pluginmanager.hasplugin("pytest_asyncio"):
        return None

    testfunction = pyfuncitem.obj
    if not inspect.iscoroutinefunction(testfunction):
        return None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # Nested event loops are not supported; defer to pytest's default handling.
        return None

    kwargs = {arg: pyfuncitem.funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}

    coroutine = testfunction(**kwargs)
    run_coroutine(coroutine, loop=kwargs.get("event_loop"))
    return True
