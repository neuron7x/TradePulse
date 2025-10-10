"""Tests for automated validation of mypy override comments."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_mypy_overrides import OverrideViolation, check_mypy_overrides


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "mypy.ini"
    path.write_text(contents, encoding="utf-8")
    return path


def test_check_mypy_overrides_allows_todo_comments(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[mypy-demo.*]
# TODO(#1): track migration
ignore_errors = True

[mypy-other]
ignore_missing_imports = True  # TODO(#2): third-party package lacks stubs
""".strip(),
    )
    check_mypy_overrides(config)


def test_check_mypy_overrides_rejects_missing_comments(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        """
[mypy-demo]
ignore_errors = True
""".strip(),
    )
    with pytest.raises(OverrideViolation):
        check_mypy_overrides(config)
