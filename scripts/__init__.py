"""Utility scripts bundled with the TradePulse repository.

This package centralises shared helpers that keep the standalone command line
utilities under :mod:`scripts` consistent.  The module intentionally remains
lightweight so importing it has no side-effects; richer functionality is
implemented in :mod:`scripts.runtime` and consumed on demand by the individual
tools and tests.
"""

# SPDX-License-Identifier: MIT

from .runtime import (  # noqa: F401 - re-export for convenience
    ArtifactSpec,
    ParameterSpec,
    SchemaRef,
    ScriptRegistry,
    ScriptRunner,
    ScriptRunResult,
)

__all__ = [
    "ArtifactSpec",
    "ParameterSpec",
    "SchemaRef",
    "ScriptRegistry",
    "ScriptRunner",
    "ScriptRunResult",
]
