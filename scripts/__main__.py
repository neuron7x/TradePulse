"""Entry point to allow ``python -m scripts`` execution."""
from __future__ import annotations

# SPDX-License-Identifier: MIT
from .cli import main

if __name__ == "__main__":  # pragma: no cover - module execution
    raise SystemExit(main())
