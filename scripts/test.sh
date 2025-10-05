#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

echo "[test] Python unit tests (pytest) in domains/*/*/tests"
if ! python -c "import pytest" 2>/dev/null; then
  echo "pytest not found, installing..." >&2
  python -m pip install --user pytest
fi

found=0
shopt -s nullglob
for d in domains/*/*/tests; do
  if [ -d "$d" ]; then
    found=1
    echo "==> $d"
    pytest -q "$d" || exit $?
  fi
done
shopt -u nullglob

if [ $found -eq 0 ]; then
  echo "No tests found under domains/*/*/tests — OK for early-stage repo."
fi

echo "[test] Node unit tests (if any)"
if command -v node >/dev/null 2>&1; then
  if [ -f "domains/ui/dashboard/tests/test.js" ]; then
    node domains/ui/dashboard/tests/test.js
  fi
else
  echo "node not found, skipping JS tests"
fi

echo "[test] Done."
