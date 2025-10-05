#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

echo "[lint] Python (ruff)"
if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff not found, installing locally..." >&2
  python -m pip install --user ruff
fi
ruff check .

echo "[lint] buf (protobuf)"
if command -v buf >/dev/null 2>&1; then
  buf lint || true
else
  echo "buf not found, skipping buf lint"
fi

echo "[lint] Done."
