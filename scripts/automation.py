#!/usr/bin/env python3
"""Unified automation entrypoint for repository scripts.

This module centralises logic for executing tox/nox session matrices,
rendering shell completions, managing configuration schema versions and
migrations, and providing developer-friendly diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import locale
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "script_runner.yml"
MIGRATION_DIR = REPO_ROOT / "configs" / "migrations"
AUDIT_LOG_PATH = REPO_ROOT / "reports" / "script_audit.log"
CURRENT_SCHEMA_VERSION = 1

LOGGER = logging.getLogger("scripts.automation")
AUDIT_LOGGER = logging.getLogger("scripts.automation.audit")


class ScriptError(RuntimeError):
    """Base error for automation failures."""


class DependencyError(ScriptError):
    """Raised when an external dependency is unavailable."""


class ConfigError(ScriptError):
    """Raised when the configuration file cannot be loaded or migrated."""


def _ensure_utf8_locale() -> None:
    """Force a UTF-8 locale so subprocesses behave consistently."""

    preferred_locales = ("C.UTF-8", "en_US.UTF-8", "en_GB.UTF-8")
    for loc in preferred_locales:
        try:
            locale.setlocale(locale.LC_ALL, loc)
        except locale.Error:
            continue
        else:
            os.environ.setdefault("LC_ALL", loc)
            os.environ.setdefault("LANG", loc)
            LOGGER.debug("Locale initialised to %s", loc)
            break
    else:
        LOGGER.warning(
            "Could not set an explicit UTF-8 locale; falling back to default"
        )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    AUDIT_LOGGER.setLevel(logging.INFO)
    AUDIT_LOGGER.addHandler(audit_handler)


@dataclass
class AuditEvent:
    """Structured audit event for reproducibility."""

    action: str
    details: Mapping[str, Any]

    def to_json(self) -> str:
        timestamp = (
            _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
        )
        payload = {
            "timestamp": timestamp,
            "user": getpass.getuser(),
            "action": self.action,
            "details": self.details,
        }
        return json.dumps(payload, ensure_ascii=False)


def record_audit(action: str, **details: Any) -> None:
    AUDIT_LOGGER.info(AuditEvent(action=action, details=details).to_json())


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return data


def _deep_update(mapping: MutableMapping[str, Any], other: Mapping[str, Any]) -> None:
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(mapping.get(key), MutableMapping):
            _deep_update(mapping[key], value)  # type: ignore[index]
        else:
            mapping[key] = value  # type: ignore[index]


def _run_migration_step(
    current: int, target: int, payload: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    migration_path = MIGRATION_DIR / f"{current}_to_{target}.yml"
    if not migration_path.exists():
        raise ConfigError(
            f"Missing migration file for {current}->{target}: {migration_path}."
            " Create the migration to keep environments aligned."
        )
    with migration_path.open("r", encoding="utf-8") as handle:
        migration = yaml.safe_load(handle) or {}
    if not isinstance(migration, Mapping):
        raise ConfigError(f"Migration file must contain a mapping: {migration_path}")

    new_payload = dict(payload)
    additions = migration.get("add", {})
    if isinstance(additions, Mapping):
        _deep_update(new_payload, additions)
    removals = migration.get("remove", {})
    if isinstance(removals, Mapping):
        for key in removals:
            new_payload.pop(key, None)
    return new_payload


def load_config(
    auto_migrate: bool = True, persist: bool = False
) -> Tuple[Dict[str, Any], int]:
    """Load the script runner configuration applying migrations if required."""

    data = _load_yaml(CONFIG_PATH)
    version = int(data.get("version", 0))

    if version > CURRENT_SCHEMA_VERSION:
        raise ConfigError(
            f"Configuration version {version} is newer than supported {CURRENT_SCHEMA_VERSION}."
            " Please upgrade the tooling before proceeding."
        )

    original_version = version
    payload: MutableMapping[str, Any] = dict(data)
    if auto_migrate:
        while version < CURRENT_SCHEMA_VERSION:
            payload = _run_migration_step(version, version + 1, payload)
            version += 1
        payload["version"] = CURRENT_SCHEMA_VERSION
        if persist and version != original_version:
            with CONFIG_PATH.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            record_audit(
                "config.migrated", from_version=original_version, to_version=version
            )
    return dict(payload), original_version


def ensure_dependency(binary: str, install_hint: str) -> None:
    if shutil.which(binary):
        return
    raise DependencyError(
        f"Required dependency '{binary}' is not available. Hint: {install_hint}"
    )


def run_subprocess(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    LOGGER.debug("Executing %s", " ".join(command))
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise DependencyError(f"Command '{command[0]}' is not available") from exc


def handle_session(args: argparse.Namespace) -> int:
    config, _ = load_config()
    matrices = config.get("matrices", {})
    if args.matrix not in matrices:
        raise ConfigError(f"Matrix '{args.matrix}' is not defined in the configuration")
    matrix: Mapping[str, Any] = matrices[args.matrix]

    tool = args.tool
    if tool == "auto":
        tool = matrix.get("tool", "nox")
    if tool not in {"tox", "nox"}:
        raise ConfigError(f"Unsupported tool '{tool}' for matrix {args.matrix}")

    sessions_key = "sessions" if tool == "nox" else "environments"
    entries: Iterable[Mapping[str, Any]] = matrix.get(sessions_key, [])
    if not entries:
        raise ConfigError(
            f"Matrix '{args.matrix}' does not provide any '{sessions_key}' entries for tool {tool}"
        )

    if args.dry_run:
        if not shutil.which(tool):
            LOGGER.warning(
                "%s not found; install it before running without --dry-run", tool
            )
    else:
        ensure_dependency(tool, f"python -m pip install {tool}")

    for entry in entries:
        name = entry.get("name")
        if not name:
            raise ConfigError(f"Matrix entry missing name in matrix '{args.matrix}'")
        description = entry.get("description", "")
        command = [tool]
        if tool == "nox":
            command.extend(["-s", name])
        else:
            command.extend(["-e", name])
        record_audit(
            "session.run",
            matrix=args.matrix,
            tool=tool,
            session=name,
            description=description,
        )
        LOGGER.info(
            "Running %s (%s)", " ".join(command), description or "no description"
        )
        if args.dry_run:
            continue
        run_subprocess(command, cwd=REPO_ROOT)
    return 0


def generate_completion(args: argparse.Namespace) -> int:
    shells = {"bash", "zsh", "fish"}
    shell = args.shell
    if shell not in shells:
        raise ScriptError(
            f"Unsupported shell '{shell}'. Supported shells: {', '.join(sorted(shells))}"
        )

    commands = {
        "session": ["--matrix", "--tool", "--dry-run"],
        "completion": ["--shell", "--binary"],
        "format": ["--check"],
        "units": [],
        "migrate": [],
    }
    binary_name = args.binary or "tradepulse-automation"
    fn_name = f"_{binary_name.replace('-', '_')}"

    if shell in {"bash", "zsh"}:
        lines: List[str] = [
            f"{fn_name}()",
            "{",
            "    local cur prev",
            "    COMPREPLY=()",
            '    cur="${COMP_WORDS[COMP_CWORD]}"',
            '    prev="${COMP_WORDS[COMP_CWORD-1]}"',
            "    if [ $COMP_CWORD -eq 1 ]; then",
            '        COMPREPLY=( $(compgen -W "{}" -- "$cur") )'.format(
                " ".join(commands)
            ),
            "        return 0",
            "    fi",
            '    case "${COMP_WORDS[1]}" in',
        ]
        for cmd, options in commands.items():
            option_list = " ".join(options)
            lines.append(f"    {cmd})")
            if option_list:
                lines.append(
                    '        COMPREPLY=( $(compgen -W "{}" -- "$cur") )'.format(
                        option_list
                    )
                )
                lines.append("        return 0")
            lines.append("        ;;")
        lines.extend(["    esac", "}"])
        if shell == "bash":
            lines.append(f"complete -F {fn_name} {binary_name}")
        else:
            lines.append(f"compdef {fn_name} {binary_name}")
        script = "\n".join(lines) + "\n"
    else:  # fish
        command_list = " ".join(commands)
        lines = [
            f"complete -c {binary_name} -n 'not __fish_seen_subcommand_from {command_list}' -a '{command_list}'"
        ]
        for cmd, options in commands.items():
            for option in options:
                lines.append(
                    f"complete -c {binary_name} -n '__fish_seen_subcommand_from {cmd}' -l {option.lstrip('-')}"
                )
        script = "\n".join(lines) + "\n"

    sys.stdout.write(script)
    return 0


def run_format(args: argparse.Namespace) -> int:
    ensure_dependency("black", "python -m pip install black")
    ensure_dependency("shfmt", "go install mvdan.cc/sh/v3/cmd/shfmt@latest")

    config, _ = load_config()
    format_cfg = (
        config.get("format", {}) if isinstance(config.get("format"), Mapping) else {}
    )
    python_targets = [
        REPO_ROOT / target for target in format_cfg.get("python", ["scripts"])
    ]
    shell_targets = [
        REPO_ROOT / target
        for target in format_cfg.get("shell", ["scripts", "deploy", "tools"])
    ]

    record_audit(
        "format.run",
        check=args.check,
        python=len(python_targets),
        shell=len(shell_targets),
    )

    existing_python_targets = [str(path) for path in python_targets if path.exists()]
    if existing_python_targets:
        LOGGER.info(
            "Running black (%s mode) on %s",
            "check" if args.check else "write",
            existing_python_targets,
        )
        black_cmd = ["black"]
        if args.check:
            black_cmd.append("--check")
        black_cmd.extend(existing_python_targets)
        run_subprocess(black_cmd)
    else:
        LOGGER.info("Skipping black run; no configured python targets exist")

    existing_shell_targets = [str(path) for path in shell_targets if path.exists()]
    if existing_shell_targets:
        LOGGER.info(
            "Running shfmt (%s mode) on %s",
            "check" if args.check else "write",
            existing_shell_targets,
        )
        shfmt_cmd = ["shfmt", "-i", "2", "-sr", "-ci"]
        if args.check:
            shfmt_cmd.append("-d")
        else:
            shfmt_cmd.append("-w")
        shfmt_cmd.extend(existing_shell_targets)
        run_subprocess(shfmt_cmd)
    else:
        LOGGER.info("Skipping shfmt run; no configured shell targets exist")
    return 0


def validate_units(args: argparse.Namespace) -> int:
    config, _ = load_config()
    conversions = config.get("units", {}).get("conversions", [])
    errors: List[str] = []
    for entry in conversions:
        if not isinstance(entry, Mapping):
            errors.append(f"Invalid conversion entry: {entry}")
            continue
        src = entry.get("from")
        dst = entry.get("to")
        factor = entry.get("factor")
        if not src or not dst:
            errors.append(f"Conversion entry missing 'from' or 'to': {entry}")
        if not isinstance(factor, (int, float)) or factor <= 0:
            errors.append(f"Invalid conversion factor for {src}->{dst}: {factor}")
    if errors:
        raise ScriptError("; ".join(errors))
    LOGGER.info("Validated %d unit conversions", len(conversions))
    record_audit("units.validated", count=len(conversions))
    return 0


def migrate_config(args: argparse.Namespace) -> int:
    config, _ = load_config(auto_migrate=False)
    current_version = int(config.get("version", 0))
    if current_version >= CURRENT_SCHEMA_VERSION:
        LOGGER.info("Configuration already at version %s", current_version)
        return 0

    payload = dict(config)
    version = current_version
    while version < CURRENT_SCHEMA_VERSION:
        payload = _run_migration_step(version, version + 1, payload)
        version += 1
    payload["version"] = CURRENT_SCHEMA_VERSION
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    record_audit(
        "config.migrated",
        from_version=current_version,
        to_version=CURRENT_SCHEMA_VERSION,
    )
    LOGGER.info(
        "Migrated configuration from %s to %s", current_version, CURRENT_SCHEMA_VERSION
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradePulse unified automation CLI")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )

    subparsers = parser.add_subparsers(dest="command")

    session_parser = subparsers.add_parser(
        "session", help="Run a configured tox/nox matrix"
    )
    session_parser.add_argument(
        "--matrix", default="default", help="Matrix name to execute"
    )
    session_parser.add_argument(
        "--tool",
        choices=["auto", "tox", "nox"],
        default="auto",
        help="Override automation tool",
    )
    session_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    session_parser.set_defaults(func=handle_session)

    completion_parser = subparsers.add_parser(
        "completion", help="Generate shell completion script"
    )
    completion_parser.add_argument(
        "--shell", required=True, help="Shell type (bash/zsh/fish)"
    )
    completion_parser.add_argument("--binary", help="Binary/alias name to generate for")
    completion_parser.set_defaults(func=generate_completion)

    format_parser = subparsers.add_parser("format", help="Run code formatters")
    format_parser.add_argument(
        "--check", action="store_true", help="Run in check (non-modifying) mode"
    )
    format_parser.set_defaults(func=run_format)

    units_parser = subparsers.add_parser("units", help="Validate unit conversions")
    units_parser.set_defaults(func=validate_units)

    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate script configuration schema"
    )
    migrate_parser.set_defaults(func=migrate_config)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_utf8_locale()
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except DependencyError as exc:
        LOGGER.error("%s", exc)
        LOGGER.info("Use the hinted installation command and retry.")
        return 1
    except ConfigError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 2
    except ScriptError as exc:
        LOGGER.error("%s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
