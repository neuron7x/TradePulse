#!/usr/bin/env bash
# Wrapper that prepares the runtime environment before launching the TradePulse API server.

set -euo pipefail

APP_HOME=${APP_HOME:-/opt/tradepulse/current}
VENV_DIR=${VENV_DIR:-/opt/tradepulse/venv}
HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
LOG_LEVEL=${UVICORN_LOG_LEVEL:-info}

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS=$(nproc --ignore=1 2>/dev/null || nproc)
else
  DEFAULT_WORKERS=2
fi
WORKERS=${UVICORN_WORKERS:-${DEFAULT_WORKERS}}

if [[ ! -x "${VENV_DIR}/bin/uvicorn" ]]; then
  echo "uvicorn binary not found under ${VENV_DIR}/bin. Ensure the virtualenv is provisioned." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TRADEPULSE_RUNTIME_ENV=${TRADEPULSE_RUNTIME_ENV:-production}

cd "${APP_HOME}"

exec "${VENV_DIR}/bin/uvicorn" application.api.service:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  --proxy-headers \
  --log-level "${LOG_LEVEL}"
