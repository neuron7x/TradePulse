from __future__ import annotations

# SPDX-License-Identifier: MIT

import logging
from argparse import _SubParsersAction
from pathlib import Path

from scripts import resilient_data_sync
from scripts.commands.base import CommandError, register
from scripts.runtime import EXIT_CODES, compute_checksum
from scripts.security_audit import scan_paths

LOGGER = logging.getLogger(__name__)


def build_parser(subparsers: _SubParsersAction[object]) -> None:
    parser = subparsers.add_parser(
        "security",
        help="Run repository security hygiene checks",
    )
    parser.set_defaults(command="security", handler=handle)
    parser.add_argument(
        "--skip-secret-scan",
        action="store_true",
        help="Skip scanning the repository for accidentally committed secrets.",
    )
    parser.add_argument(
        "--skip-backup-drill",
        action="store_true",
        help="Skip running the backup/restore smoke test.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="Optional list of paths to scan for secrets (defaults to repository root).",
    )


def _run_secret_scan(paths: list[Path]) -> None:
    LOGGER.info("Scanning %d path(s) for sensitive credentials…", len(paths))
    findings = scan_paths(paths)
    if findings:
        for finding in findings:
            LOGGER.error(
                "Potential secret detected: %s:%d -> %s (%s)",
                finding.path,
                finding.line,
                finding.pattern,
                finding.match,
            )
        raise CommandError(
            "Potential secrets detected. Review the log output above before continuing."
        )
    LOGGER.info("No high-risk secrets detected in the scanned paths.")


def _run_backup_drill(tmp_root: Path) -> None:
    LOGGER.info("Executing backup integrity drill via resilient_data_sync…")
    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "sample.csv"
    if not sample.exists():
        raise CommandError("sample.csv not found – unable to exercise backup drill.")

    checksum = compute_checksum(sample)
    exit_code = resilient_data_sync.main(
        [
            str(sample),
            "--artifact-root",
            str(tmp_root),
            "--checksum",
            f"{sample}={checksum}",
        ]
    )

    if exit_code != EXIT_CODES["success"]:
        raise CommandError("Backup drill reported failures. Inspect logs for details.")
    LOGGER.info("Backup drill completed successfully.")


@register("security")
def handle(args: object) -> int:
    namespace = getattr(args, "__dict__", args)
    skip_secret_scan = bool(namespace.get("skip_secret_scan", False))
    skip_backup_drill = bool(namespace.get("skip_backup_drill", False))
    raw_paths = namespace.get("paths") or []
    repo_root = Path.cwd()
    paths = [repo_root / Path(path) for path in raw_paths] if raw_paths else [repo_root]

    if not skip_secret_scan:
        _run_secret_scan(paths)
    else:
        LOGGER.info("Skipping secret scan per CLI flag.")

    if not skip_backup_drill:
        artifacts_root = repo_root / "reports" / "security"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        _run_backup_drill(artifacts_root)
    else:
        LOGGER.info("Skipping backup drill per CLI flag.")

    LOGGER.info("Security checks finished without errors.")
    return 0
