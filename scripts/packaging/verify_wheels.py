"""Utility to verify produced wheels advertise manylinux/musllinux compatibility."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REQUIRED_TAGS = ("manylinux", "musllinux")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path, help="Directory containing built wheels")
    return parser.parse_args()


def auditwheel_tags(wheel: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    args = parse_args()
    wheels = sorted(args.wheel_dir.glob("*.whl"))
    if not wheels:
        print(f"No wheels found in {args.wheel_dir}", file=sys.stderr)
        return 1
    failures: list[str] = []
    for wheel in wheels:
        if not any(tag in wheel.name for tag in REQUIRED_TAGS):
            failures.append(f"{wheel.name} does not advertise a manylinux/musllinux tag in its filename")
            continue
        output = auditwheel_tags(wheel)
        if not any(tag in output for tag in REQUIRED_TAGS):
            failures.append(f"auditwheel show output for {wheel.name} does not contain an expected compatibility tag")
    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 2
    for wheel in wheels:
        print(f"Validated wheel compatibility tags for {wheel.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
