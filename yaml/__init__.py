"""Lightweight YAML subset parser for TradePulse.

This module implements a tiny subset of the PyYAML API used within the
project.  It intentionally supports only the features required by our
configuration files (nested mappings, inline lists, and scalars).  The goal
is to keep tests self-contained without introducing the PyYAML dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


class YAMLError(Exception):
    """Generic YAML parsing error."""


def safe_load(stream: Any) -> Any:
    """Parse *stream* into native Python objects.

    The parser understands the limited YAML surface used across the repository:
    nested mappings defined by indentation, inline lists (``[1, 2, 3]``), and
    scalars such as integers, floats, booleans, and strings.  Multi-document
    streams, anchors, complex tags, and custom constructors are intentionally
    unsupported to keep the implementation compact and deterministic.
    """

    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = stream

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):  # pragma: no cover - defensive programming
        raise TypeError("safe_load expects a string, bytes, or file-like object")

    parser = _SimpleYAMLParser()
    return parser.parse(text)


# ---------------------------------------------------------------------------
# Internal parser implementation

_BOOLEAN_TRUE = {"true", "yes", "on"}
_BOOLEAN_FALSE = {"false", "no", "off"}
_NULL_VALUES = {"null", "none", "~"}


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return ""

    lower = token.lower()
    if lower in _BOOLEAN_TRUE:
        return True
    if lower in _BOOLEAN_FALSE:
        return False
    if lower in _NULL_VALUES:
        return None

    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]

    try:
        if any(ch in lower for ch in (".", "e")):
            return float(token)
        return int(token, 0)
    except ValueError:
        pass

    return token


def _parse_list(value: str) -> List[Any]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in inner:
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if char == "[":
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return [_parse_scalar(part) if not part.startswith("[") else _parse_list(part) for part in parts]


@dataclass
class _StackEntry:
    indent: int
    container: Any
    parent: "_StackEntry | None"
    key: str | None = None

    def ensure_list(self) -> List[Any]:
        if isinstance(self.container, list):
            return self.container
        if isinstance(self.container, dict):
            if self.parent is None or self.key is None:
                raise YAMLError("List item without a preceding key")
            new_list: List[Any] = []
            self.parent.container[self.key] = new_list
            self.container = new_list
            return new_list
        raise YAMLError("Invalid container for list items")


class _SimpleYAMLParser:
    def parse(self, text: str) -> Any:
        root: dict[str, Any] = {}
        stack: List[_StackEntry] = [_StackEntry(indent=-1, container=root, parent=None, key=None)]

        for raw_line in text.splitlines():
            line, *_ = raw_line.split("#", 1)
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            while stack and indent <= stack[-1].indent:
                stack.pop()
            if not stack:
                raise YAMLError("Invalid indentation structure in YAML stream")

            current = stack[-1]

            if stripped.startswith("- "):
                list_container = current.ensure_list()
                item_value = stripped[2:].strip()
                if item_value.endswith(":"):
                    key = item_value[:-1].strip()
                    item_dict: dict[str, Any] = {}
                    list_container.append(item_dict)
                    stack.append(_StackEntry(indent=indent, container=item_dict, parent=current, key=None))
                else:
                    value = self._parse_value(item_value)
                    list_container.append(value)
                continue

            if ":" not in stripped:
                raise YAMLError(f"Unable to parse line: {raw_line!r}")

            key, remainder = stripped.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()

            if remainder:
                value = self._parse_value(remainder)
                if isinstance(current.container, list):
                    current.container.append({key: value})
                else:
                    current.container[key] = value
            else:
                new_container: dict[str, Any] = {}
                if isinstance(current.container, list):
                    current.container.append({key: new_container})
                    parent_entry = _StackEntry(indent=indent, container=new_container, parent=current, key=None)
                    stack.append(parent_entry)
                else:
                    current.container[key] = new_container
                    stack.append(_StackEntry(indent=indent, container=new_container, parent=current, key=key))

        return root

    def _parse_value(self, value: str) -> Any:
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            return _parse_list(value)
        return _parse_scalar(value)


__all__ = ["safe_load", "YAMLError"]
