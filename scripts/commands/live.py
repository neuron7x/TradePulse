# SPDX-License-Identifier: MIT
"""Run the production live execution loop using TOML configuration."""

from __future__ import annotations

import logging
from argparse import _SubParsersAction
from pathlib import Path
from typing import Sequence

from interfaces.live_runner import LiveTradingRunner
from scripts.commands.base import register

LOGGER = logging.getLogger(__name__)


def build_parser(subparsers: _SubParsersAction[object]) -> None:
    parser = subparsers.add_parser("live", help=__doc__)
    parser.set_defaults(command="live", handler=handle)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the live trading TOML configuration file (defaults to configs/live/default.toml).",
    )
    parser.add_argument(
        "--venue",
        dest="venues",
        action="append",
        default=None,
        help="Restrict execution to the specified venue (can be supplied multiple times).",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Override the state directory used by the live loop.",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Skip reconciliation and treat this launch as a cold start.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on the provided port.",
    )


@register("live")
def handle(args: object) -> int:
    namespace = getattr(args, "__dict__", args)
    config_path: Path | None = namespace.get("config")
    venues: Sequence[str] | None = namespace.get("venues")
    state_dir: Path | None = namespace.get("state_dir")
    cold_start: bool = bool(namespace.get("cold_start", False))
    metrics_port: int | None = namespace.get("metrics_port")

    runner = LiveTradingRunner(
        config_path,
        venues=venues,
        state_dir_override=state_dir,
        metrics_port=metrics_port,
    )

    LOGGER.info(
        "Launching live trading command",
        extra={
            "event": "scripts.live.start",
            "config": str(runner.config_path),
            "venues": list(runner.connectors.keys()),
            "cold_start": cold_start,
        },
    )
    runner.run(cold_start=cold_start)
    return 0


__all__ = ["build_parser", "handle"]
