# SPDX-License-Identifier: MIT

# === FPM-A (Fractal Project Method) integration ===
.PHONY: fpma-graph fpma-check lock build-package publish-package clean-dist
fpma-graph:
	python3 tools/fpma_runner.py graph

fpma-check:
	python3 tools/fpma_runner.py check

.PHONY: lock
lock:
        python -m pip install --upgrade pip
        python -m pip install pip-tools uv
        pip-compile pyproject.toml --resolver=backtracking --output-file=requirements.lock
        pip-compile requirements-dev.txt --resolver=backtracking --output-file=requirements-dev.lock
        uv pip compile pyproject.toml --output-file uv.lock

.PHONY: build-package
build-package: clean-dist
	python -m build --sdist --wheel --outdir dist

.PHONY: publish-package
publish-package: build-package
	twine check dist/*
	twine upload dist/*

.PHONY: clean-dist
clean-dist:
	rm -rf dist build *.egg-info

.PHONY: generate
generate:
	buf generate
	PYTHONPATH=. python tools/schema/generate_event_types.py
