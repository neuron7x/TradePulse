#!/usr/bin/env bash
set -euo pipefail

# Weekly dependency refresh routine for TradePulse.
# Usage: ./scripts/dep_update.sh [--upgrade-package package==version]
# Requires Python 3.10+ with pip-tools installed (install on-demand below).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python3}
PIP_COMPILE_ARGS=(--generate-hashes --resolver=backtracking --allow-unsafe -c constraints.txt)

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python binary '${PYTHON_BIN}' not found" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install --upgrade pip "pip-tools>=7.5.1" "pip-audit>=2.9.0" "cyclonedx-bom>=4.5.0"

function compile_lock() {
  local output=$1
  local source=$2
  shift 2
  "${PYTHON_BIN}" -m piptools compile "${PIP_COMPILE_ARGS[@]}" "$@" --output-file="${output}" "${source}"
}

compile_lock requirements.txt requirements.in "$@"
compile_lock dev.txt dev.in "$@"

printf '# Managed via pip-tools; see requirements.in for sources.\n-r requirements.txt\n' > requirements.lock
printf '# Managed via pip-tools; see dev.in for sources.\n-r dev.txt\n' > requirements-dev.txt
cp requirements-dev.txt requirements-dev.lock

"${PYTHON_BIN}" -m pip_audit -r requirements.txt -r dev.txt --progress-spinner off

cyclonedx-bom -o sbom/cyclonedx-sbom.json -e requirements.txt dev.txt

if command -v git >/dev/null 2>&1; then
  git status --short requirements.in dev.in constraints.txt requirements.txt dev.txt requirements.lock requirements-dev.txt requirements-dev.lock sbom/cyclonedx-sbom.json
fi

echo "✅ Dependency refresh complete"
