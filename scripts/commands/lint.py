"""Linting subcommand implementation."""
from __future__ import annotations

# SPDX-License-Identifier: MIT

import logging
import shutil
from argparse import _SubParsersAction

from scripts.commands.base import CommandError, register, run_subprocess

LOGGER = logging.getLogger(__name__)


def _has_tool(tool: str) -> bool:
    return shutil.which(tool) is not None


def build_parser(subparsers: _SubParsersAction[object]) -> None:
    parser = subparsers.add_parser("lint", help="Run static analysis tooling")
    parser.set_defaults(command="lint", handler=handle)
    parser.add_argument(
        "--skip-buf",
        action="store_true",
        help="Skip protobuf linting even if buf is available.",
    )


@register("lint")
def handle(args: object) -> int:
    namespace = getattr(args, "__dict__", args)
    skip_buf = namespace.get("skip_buf", False)

    if not _has_tool("ruff"):
        raise CommandError("ruff is required but was not found in PATH. Install it to continue.")

    LOGGER.info("Running ruff lint checks…")
    run_subprocess(["ruff", "check", "."])

    if not skip_buf and _has_tool("buf"):
        LOGGER.info("Running buf lint checks…")
        run_subprocess(["buf", "lint"], check=False)
    elif not skip_buf:
        LOGGER.info("buf executable not available – skipping protobuf linting.")

    LOGGER.info("Lint checks completed successfully.")
    return 0

