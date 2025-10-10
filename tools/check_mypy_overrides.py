"""Validate that mypy configuration overrides reference tracking issues."""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

TARGET_KEYS = ("ignore_errors", "ignore_missing_imports")
TODO_PATTERN = re.compile(r"TODO\(#\d+\)")


class OverrideViolation(Exception):
    """Raised when a mypy override does not include a tracking comment."""


def _has_todo_comment(line: str) -> bool:
    return "#" in line and bool(TODO_PATTERN.search(line))


def _previous_comment(lines: Sequence[str], index: int) -> str | None:
    for offset in range(index - 1, -1, -1):
        candidate = lines[offset].strip()
        if not candidate:
            continue
        if candidate.startswith("["):
            return None
        if candidate.startswith("#"):
            return candidate
        return None
    return None


def check_mypy_overrides(path: Path) -> None:
    """Ensure every override has an inline or preceding TODO issue comment."""

    contents = path.read_text(encoding="utf-8").splitlines()
    violations: list[str] = []

    for idx, line in enumerate(contents):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in TARGET_KEYS:
            continue
        if _has_todo_comment(stripped):
            continue
        previous = _previous_comment(contents, idx)
        if previous is not None and TODO_PATTERN.search(previous):
            continue
        violations.append(f"Line {idx + 1}: override '{key}' missing TODO issue reference")

    if violations:
        message = "\n".join(violations)
        raise OverrideViolation(message)


def main(argv: Iterable[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    target = Path(args[0]) if args else Path("mypy.ini")
    try:
        check_mypy_overrides(target)
    except FileNotFoundError as exc:  # pragma: no cover - guard for CI misconfiguration
        print(f"error: unable to read mypy config: {exc}", file=sys.stderr)
        return 2
    except OverrideViolation as exc:
        print("error: mypy overrides missing TODO comments", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via command line
    sys.exit(main())
