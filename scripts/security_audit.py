"""Lightweight repository secret scanning utilities."""
from __future__ import annotations

# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator

_PLACEHOLDER_HINTS = ("changeme", "change_me", "example", "sample", "your_", "demo", "placeholder")

_DEFAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "AWS temporary access key": re.compile(r"ASIA[0-9A-Z]{16}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "GitHub personal access token": re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,48}"),
    "Stripe secret key": re.compile(r"sk_(live|test)_[0-9a-zA-Z]{24,}"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.([A-Za-z0-9_-]{10,})\.([A-Za-z0-9_-]{10,})"),
}

_EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    "reports",
    "sbom",
    ".next",
}

_EXCLUDED_FILES = {
    ".secrets.baseline",
}

_MAX_SCAN_FILE_SIZE = 1_000_000  # bytes


@dataclass(slots=True)
class SecretFinding:
    path: Path
    line: int
    pattern: str
    match: str

    def __str__(self) -> str:  # pragma: no cover - convenience for callers
        return f"{self.path}:{self.line} -> {self.pattern} ({self.match})"


def _looks_like_placeholder(candidate: str) -> bool:
    lowered = candidate.lower()
    return any(hint in lowered for hint in _PLACEHOLDER_HINTS)


def _iter_files(root: Path) -> Iterator[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        if current.is_dir():
            if current.name in _EXCLUDED_DIRS:
                continue
            for child in current.iterdir():
                stack.append(child)
        elif current.is_file():
            if current.name in _EXCLUDED_FILES:
                continue
            try:
                size = current.stat().st_size
            except OSError:
                continue
            if size > _MAX_SCAN_FILE_SIZE:
                continue
            yield current


def scan_paths(paths: Iterable[Path], patterns: dict[str, re.Pattern[str]] | None = None) -> list[SecretFinding]:
    compiled = patterns or _DEFAULT_PATTERNS
    findings: list[SecretFinding] = []

    for root in paths:
        root_path = root
        if not root_path.exists():
            continue
        for candidate in _iter_files(root_path):
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                for label, regex in compiled.items():
                    for match in regex.finditer(line):
                        snippet = match.group(0)
                        if _looks_like_placeholder(snippet):
                            continue
                        findings.append(
                            SecretFinding(
                                path=candidate,
                                line=line_number,
                                pattern=label,
                                match=snippet,
                            )
                        )
    return findings


__all__ = ["SecretFinding", "scan_paths"]
