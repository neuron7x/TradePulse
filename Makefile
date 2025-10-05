# SPDX-License-Identifier: MIT

# === FPM-A (Fractal Project Method) integration ===
.PHONY: fpma-graph fpma-check
fpma-graph:
	python3 tools/fpma_runner.py graph

fpma-check:
	python3 tools/fpma_runner.py check



.PHONY: generate
generate:
	buf generate
