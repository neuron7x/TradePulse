#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

python scripts/automation.py session --matrix default "$@"
