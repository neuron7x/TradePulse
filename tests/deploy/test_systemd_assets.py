"""Tests covering the systemd deployment assets for TradePulse."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

START_SCRIPT = Path("deploy/systemd/start-tradepulse.sh")
SERVICE_FILE = Path("deploy/systemd/tradepulse.service")


def _write_uvicorn_stub(trace_base: Path, target: Path) -> None:
    """Create an executable stub that records CLI arguments and env vars."""

    script = """#!/usr/bin/env bash
set -euo pipefail

TRACE_BASE=${TRACE_FILE:?TRACE_FILE must be set}

printf '%s\n' "$@" > "${TRACE_BASE}.args"
env | sort > "${TRACE_BASE}.env"
"""

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script)
    target.chmod(target.stat().st_mode | stat.S_IEXEC)


def _base_env() -> dict[str, str]:
    env = {"HOME": os.environ.get("HOME", ""), "LANG": os.environ.get("LANG", "C.UTF-8")}
    # Ensure bash is discoverable while allowing tests to customise the lookup path for nproc.
    env["PATH"] = os.environ.get("PATH", "/bin:/usr/bin")
    return env


def test_start_script_invokes_uvicorn_with_defaults_when_nproc_missing(tmp_path: Path) -> None:
    env = _base_env()
    env["PATH"] = "/bin"  # No nproc available, so the script uses the fallback workers value.

    app_home = tmp_path / "app"
    venv_dir = tmp_path / "venv"
    app_home.mkdir()

    uvicorn_binary = venv_dir / "bin" / "uvicorn"
    trace_base = tmp_path / "trace-default"
    env.update(
        {
            "APP_HOME": str(app_home),
            "VENV_DIR": str(venv_dir),
            "TRACE_FILE": str(trace_base),
        }
    )

    _write_uvicorn_stub(trace_base, uvicorn_binary)

    subprocess.run([str(START_SCRIPT)], env=env, check=True)

    args = (trace_base.with_suffix(".args")).read_text().splitlines()
    assert args == [
        "application.api.service:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "2",
        "--proxy-headers",
        "--log-level",
        "info",
    ]

    env_dump = (trace_base.with_suffix(".env")).read_text().splitlines()
    assert "PYTHONUNBUFFERED=1" in env_dump
    assert "TRADEPULSE_RUNTIME_ENV=production" in env_dump


def test_start_script_respects_env_overrides_and_nproc_fallback(tmp_path: Path) -> None:
    env = _base_env()

    # Provide a stub nproc that fails for --ignore=1 and succeeds otherwise to exercise the fallback.
    nproc_bin = tmp_path / "bin" / "nproc"
    nproc_script = """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--ignore=1" ]]; then
  exit 1
fi
echo 7
"""
    nproc_bin.parent.mkdir(parents=True, exist_ok=True)
    nproc_bin.write_text(nproc_script)
    nproc_bin.chmod(nproc_bin.stat().st_mode | stat.S_IEXEC)

    env["PATH"] = f"{nproc_bin.parent}:{env['PATH']}"

    app_home = tmp_path / "app"
    venv_dir = tmp_path / "venv"
    app_home.mkdir()

    uvicorn_binary = venv_dir / "bin" / "uvicorn"
    trace_base = tmp_path / "trace-overrides"

    env.update(
        {
            "APP_HOME": str(app_home),
            "VENV_DIR": str(venv_dir),
            "TRACE_FILE": str(trace_base),
            "API_HOST": "127.0.0.1",
            "API_PORT": "9200",
            "UVICORN_LOG_LEVEL": "debug",
            "TRADEPULSE_RUNTIME_ENV": "staging",
        }
    )

    _write_uvicorn_stub(trace_base, uvicorn_binary)

    subprocess.run([str(START_SCRIPT)], env=env, check=True)

    args = (trace_base.with_suffix(".args")).read_text().splitlines()
    assert args == [
        "application.api.service:app",
        "--host",
        "127.0.0.1",
        "--port",
        "9200",
        "--workers",
        "7",
        "--proxy-headers",
        "--log-level",
        "debug",
    ]

    env_dump = (trace_base.with_suffix(".env")).read_text().splitlines()
    assert "TRADEPULSE_RUNTIME_ENV=staging" in env_dump


def test_start_script_errors_when_uvicorn_binary_missing(tmp_path: Path) -> None:
    env = _base_env()
    env["PATH"] = "/bin"

    app_home = tmp_path / "app"
    venv_dir = tmp_path / "venv"
    app_home.mkdir()

    env.update({"APP_HOME": str(app_home), "VENV_DIR": str(venv_dir)})

    result = subprocess.run(
        [str(START_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "uvicorn binary not found" in result.stderr


def test_tradepulse_service_unit_configuration() -> None:
    content = SERVICE_FILE.read_text().splitlines()

    def has_directive(key: str, value: str) -> bool:
        target = f"{key}={value}"
        return any(line.strip() == target for line in content)

    assert has_directive("Description", "TradePulse FastAPI application server")
    assert has_directive("ExecStart", "/usr/local/bin/start-tradepulse.sh")
    assert has_directive("EnvironmentFile", "-/etc/tradepulse/tradepulse.env")
    assert has_directive("Restart", "always")
    assert has_directive("RestartSec", "5s")

