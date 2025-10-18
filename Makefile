# SPDX-License-Identifier: MIT

# === FPM-A (Fractal Project Method) integration ===
.PHONY: fpma-graph fpma-check lock build-package publish-package clean-dist \
        test\:fast test\:all test\:heavy
fpma-graph:
	python -m scripts fpma graph

fpma-check:
	python -m scripts fpma check

.PHONY: lock
lock:
python -m pip install --upgrade pip
python -m pip install pip-tools
python -m piptools compile --generate-hashes --resolver=backtracking --allow-unsafe --output-file=requirements.txt requirements.in -c constraints.txt
python -m piptools compile --generate-hashes --resolver=backtracking --allow-unsafe --output-file=dev.txt dev.in -c constraints.txt
printf '# Managed via pip-tools; see requirements.in for sources.\n-r requirements.txt\n' > requirements.lock
printf '# Managed via pip-tools; see dev.in for sources.\n-r dev.txt\n' > requirements-dev.txt
cp requirements-dev.txt requirements-dev.lock

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

.PHONY: scripts-lint scripts-test scripts-gen-proto scripts-dev-up scripts-dev-down
scripts-lint:
	python -m scripts lint

scripts-test:
	python -m scripts test

scripts-gen-proto:
	python -m scripts gen-proto

scripts-dev-up:
	python -m scripts dev-up

scripts-dev-down:
	python -m scripts dev-down

.PHONY: mutation-test
mutation-test:
	mutmut run --use-coverage
	mutmut results

.PHONY: sbom supply-chain-verify
sbom:
	python -m scripts supply-chain generate-sbom --include-dev --output sbom/cyclonedx-sbom.json

supply-chain-verify:
	python -m scripts supply-chain verify --include-dev

.PHONY: security-audit
security-audit:
python scripts/dependency_audit.py --requirement requirements.txt --requirement dev.txt

.PHONY: test\:fast
test\:fast:
	pytest tests/ -m "not slow and not heavy_math and not nightly"

.PHONY: test\:all
test\:all:
	pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics --cov-branch --cov-report=term-missing

.PHONY: test\:heavy
test\:heavy:
	pytest tests/ -m "slow or heavy_math or nightly"

