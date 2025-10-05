#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

case "${1:-}" in
  graph)
    python3 tools/fpma_runner.py graph
    ;;
  check)
    python3 tools/fpma_runner.py check
    ;;
  *)
    echo "Usage: $0 {graph|check}" >&2
    exit 2
    ;;
esac
