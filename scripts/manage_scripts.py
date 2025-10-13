"""Administrative helper for working with TradePulse scripts.

The CLI exposes utilities that would otherwise require a grab bag of ad-hoc
shell invocations:

* ``index`` renders a Markdown catalogue of available scripts with their
  parameters and runtime expectations.
* ``run`` executes a script via :class:`scripts.runtime.ScriptRunner`, handling
  dependency checks, manifests and metrics automatically.
* ``validate`` allows individual schema checks without executing the full
  script, making it easy to gate data ingestion steps in notebooks or CI jobs.
"""

from __future__ import annotations

# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - exercised when executed as a standalone script
    from .runtime import SchemaValidationError, ScriptRegistry, ScriptRunner
except ImportError:  # pragma: no cover - fallback for ``python scripts/manage_scripts.py``
    import sys
    from pathlib import Path

    _MODULE_ROOT = Path(__file__).resolve().parent.parent
    if str(_MODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_MODULE_ROOT))
    from scripts.runtime import SchemaValidationError, ScriptRegistry, ScriptRunner


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Generate the script index")
    index_parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/scripts-index.md"),
        help="Destination file for the Markdown index (default: docs/scripts-index.md).",
    )

    run_parser = subparsers.add_parser("run", help="Execute a registered script")
    run_parser.add_argument("script", help="Name of the script to run")
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional arguments")
    run_parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="Disable automatic dependency installation.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass safety confirmation prompts for live-affecting scripts.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip executing the underlying command while still emitting manifests.",
    )
    run_parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional override for the metrics output path.",
    )
    run_parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional override for the manifest output path.",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a payload against a configured schema"
    )
    validate_parser.add_argument("script", help="Script whose schema should be used")
    validate_parser.add_argument(
        "--schema-kind",
        choices={"pandera", "jsonschema"},
        required=True,
        help="Type of schema to evaluate.",
    )
    validate_parser.add_argument(
        "--schema-index",
        type=int,
        default=0,
        help="Index of the schema reference to use (default: 0).",
    )
    validate_parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="JSON file containing the payload to validate.",
    )

    return parser.parse_args(argv)


def _cmd_index(registry: ScriptRegistry, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(registry.generate_markdown_index(), encoding="utf-8")
    print(f"Script index written to {output}")
    return 0


def _cmd_run(
    registry: ScriptRegistry,
    args: argparse.Namespace,
) -> int:
    runner = ScriptRunner(registry)
    result = runner.run(
        args.script,
        extra_args=args.args,
        auto_install_dependencies=not args.no_auto_install,
        dry_run=args.dry_run,
        force=args.force,
        metrics_path=args.metrics_output,
        manifest_path=args.manifest_output,
    )
    status = "succeeded" if result.success else "failed"
    print(
        f"Script {args.script} {status} in {result.duration_seconds:.2f}s | "
        f"metrics: {result.metrics_path} | manifest: {result.manifest_path}"
    )
    return 0 if result.success else 1


def _cmd_validate(registry: ScriptRegistry, args: argparse.Namespace) -> int:
    spec = registry.get(args.script)
    references = [
        ref for ref in spec.input_schemas if ref.kind == args.schema_kind
    ]
    if not references:
        raise SchemaValidationError(
            f"Script '{args.script}' has no schema of kind '{args.schema_kind}'"
        )
    try:
        schema_ref = references[args.schema_index]
    except IndexError as exc:
        raise SchemaValidationError(
            f"Schema index {args.schema_index} out of range for script '{args.script}'"
        ) from exc

    payload: Any = json.loads(args.payload.read_text(encoding="utf-8"))
    schema_ref.validate(payload)
    print("Payload is valid")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry = ScriptRegistry.from_path()

    if args.command == "index":
        return _cmd_index(registry, args.output)
    if args.command == "run":
        return _cmd_run(registry, args)
    if args.command == "validate":
        return _cmd_validate(registry, args)
    raise RuntimeError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
