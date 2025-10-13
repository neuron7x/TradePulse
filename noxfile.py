"""Nox sessions for TradePulse automation."""

from __future__ import annotations

import pathlib

import nox

REPO_ROOT = pathlib.Path(__file__).parent

nox.options.sessions = ["tests-3.11", "tests-3.12", "lint"]
nox.options.error_on_missing_interpreters = False


def _install_requirements(session: nox.Session) -> None:
    session.install("-r", "requirements.lock")
    session.install("-r", "requirements-dev.lock")


@nox.session(name="tests-3.11", python="3.11")
def tests_3_11(session: nox.Session) -> None:
    """Run the primary pytest suite under Python 3.11."""

    _install_requirements(session)
    session.run(
        "pytest",
        "tests/unit/",
        "tests/integration/",
        "tests/property/",
        env={"PYTHONPATH": str(REPO_ROOT)},
    )


@nox.session(name="tests-3.12", python="3.12")
def tests_3_12(session: nox.Session) -> None:
    """Run the pytest suite under Python 3.12."""

    _install_requirements(session)
    session.run(
        "pytest",
        "tests/unit/",
        "tests/integration/",
        env={"PYTHONPATH": str(REPO_ROOT)},
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run linters via ruff and mypy."""

    _install_requirements(session)
    session.run("ruff", "check", str(REPO_ROOT))
    session.run("mypy", "core")
