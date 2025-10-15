"""Container entrypoint for TradePulse runtime."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import List


def parse_umask(value: str) -> int:
    """Parse an octal umask string into an int, defaulting to 0o027."""
    value = value.strip()
    if not value:
        return 0o027
    try:
        # Accept values like "027" or "0o027"
        return int(value, 8)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid umask value: {value!r}") from exc


def main(argv: List[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    command: List[str]
    if args:
        command = args
    else:
        command = ["-m", "nfpro", "--mode", "paper"]

    env_file = os.environ.get("APP_ENV_FILE", "").strip()
    if env_file:
        path = Path(env_file)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    umask_value = parse_umask(os.environ.get("APP_UMASK", "027"))
    os.umask(umask_value)

    executable = sys.executable
    if command and not command[0].startswith("-"):
        executable = command[0]
        command = command[1:]

    if not command:
        raise SystemExit("No command provided for container entrypoint")

    os.execvp(executable, [executable, *command])


if __name__ == "__main__":
    main()
