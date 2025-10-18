"""Shared infrastructural libraries for TradePulse.

This package collects reusable infrastructure building blocks such as
persistence helpers under :mod:`libs.db` and protocol buffers under
:mod:`libs.proto`. It exists primarily to provide a concrete package root so
static type checkers can resolve modules deterministically.
"""

__all__ = ["db", "proto"]
