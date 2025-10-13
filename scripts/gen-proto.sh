#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

# Regenerate protobuf stubs using buf (Go/Python as configured in buf.gen.yaml)
if ! command -v buf > /dev/null 2>&1; then
  echo "buf is not installed. See https://buf.build/ for installation instructions." >&2
  exit 1
fi

echo "[gen-proto] Linting proto files..."
buf lint

echo "[gen-proto] Generating code from proto..."
buf generate

echo "[gen-proto] Done."
