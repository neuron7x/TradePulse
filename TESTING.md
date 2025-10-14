# Testing Guide for TradePulse

This document describes the testing strategy, coverage requirements, and instructions for running tests in the TradePulse project.

## Overview

TradePulse employs a comprehensive testing strategy that includes:

- **Unit Tests**: Validate individual functions and classes in isolation.
- **Integration Tests**: Exercise complete workflows and pipeline chains across modules.
- **Property-Based Tests**: Check invariants and properties using Hypothesis-generated data.
- **Fuzz Tests**: Stress the system with malformed and adversarial payloads.
- **End-to-End (E2E) Tests**: Reproduce realistic user journeys and CLI pipelines.
- **Performance Tests**: Test behavior with large datasets (planned).
- **Contract Tests**: Keep DTO JSON Schemas and OpenAPI documents stable for downstream consumers.
- **Data Quality Gates**: Detect duplicates, spikes, and missing values before analytical jobs run.
- **Security Tests**: Prevent sensitive data leakage in audit logs and shared fixtures.
- **UI Smoke & Accessibility**: Exercise the Next.js dashboard via Playwright and aXe rules.

For a capability-to-suite traceability view, consult the
[regression test matrix](tests/TEST_PLAN.md) which links major product behaviours to the
automated coverage that protects them.

## Test Structure

```
tests/
├── unit/                    # Module-level tests (core.*, backtest.*, execution.*, ...)
├── integration/             # Workflow-level tests (pipelines, backtests, protocol adapters)
├── property/                # Property-based suites powered by Hypothesis
├── fuzz/                    # Fuzz harnesses targeting ingestion and message handling
├── contracts/               # JSON Schema/OpenAPI compatibility suites
├── data/                    # Data quality guardrails for reference datasets
├── security/                # Logging and secret-leak safeguards
├── e2e/                     # Pytest smoke scenarios that mimic end-user flows
└── fixtures/                # Shared fixtures, builders, and reusable data

scripts/
└── smoke_e2e.py             # CLI entrypoint used by the E2E workflow
```

## Coverage Requirements

**Target Coverage: 98%** *(current CI gate: 80% while Kuramoto/Ricci suites stabilize)*

Current module coverage targets:
- `backtest/`: 100% (ACHIEVED)
- `execution/`: 100% (ACHIEVED)
- `core/agent/`: ≥95%
- `core/data/`: ≥95%
- `core/indicators/`: ≥90%
- `core/metrics/`: ≥95%
- `core/phase/`: ≥95%

Coverage is measured using `pytest-cov` and enforced in CI.

## Running Tests Locally

### Prerequisites

Install dependencies (development file includes runtime stack):
```bash
pip install -r requirements-dev.lock
```

### Running All Tests

Run the complete test suite:
```bash
pytest tests/
```

Run with branch coverage report:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics --cov-branch --cov-report=term-missing
```

### Makefile Shortcuts

The Makefile exposes convenience targets that wrap the most common pytest invocations. Use them to keep local runs aligned with CI expectations:

| Target | Description | Underlying command |
| --- | --- | --- |
| `make test:fast` | Fast feedback loop that skips heavyweight suites | `pytest tests/ -m "not slow and not heavy_math and not nightly"` |
| `make test:all` | Full coverage-enabled suite matching CI defaults | `pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics --cov-branch --cov-report=term-missing` |
| `make test:heavy` | Executes the slow, heavy math, and nightly gates | `pytest tests/ -m "slow or heavy_math or nightly"` |

Generate HTML coverage report:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics --cov-branch --cov-report=html
# Open htmlcov/index.html in browser
```

### Running Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/
```

**Integration tests only:**
```bash
pytest tests/integration/
```

**Property-based tests only:**
```bash
pytest tests/property/
```

**Fuzz tests only:**
```bash
pytest tests/fuzz/
```

**Contract tests (JSON Schema + OpenAPI):**
```bash
pytest tests/contracts/
```

**Data quality guardrails:**
```bash
pytest tests/data/
```
When preparing new sample data files you can also run the standalone analyzer:
```bash
python scripts/data_sanity.py data --pattern "**/*.csv"
```

**Security guardrails:**
```bash
pytest tests/security/
bandit -r tests/utils tests/scripts -ll
```

**E2E smoke tests (pytest harness):**
```bash
pytest tests/e2e/
```

**Cross-architecture indicator parity (CPU/GPU/ARM simulacrum):**
```bash
pytest -m arm tests/performance/test_indicator_portability.py
```

**Performance regression benchmarks (fail on budget overruns):**
```bash
pytest tests/performance/test_indicator_benchmarks.py --benchmark-enable
```

**Memory pressure guardrails for large indicator windows:**
```bash
pytest tests/performance/test_memory_regression.py
```

**Heavy-math validation gate:**
```bash
pytest -m heavy_math tests/unit/config/test_heavy_math_jobs.py
```

**UI smoke & accessibility (Playwright):**
```bash
cd apps/web
npm ci
npm run build
npx playwright test --config=playwright.config.ts
```
The Playwright project automatically runs aXe scans and fails the run if critical or serious accessibility violations are detected.

### Running Specific Test Files or Functions

Run a specific test file:
```bash
pytest tests/unit/test_agents.py
```

Run a specific test function:
```bash
pytest tests/unit/test_agents.py::test_strategy_mutation_changes_numeric_parameters
```

Run tests matching a pattern:
```bash
pytest tests/ -k "backtest"
```

### Running the E2E Pipeline Script

The CLI entrypoint mirrors the CI smoke workflow and is suitable for local regression checks:

```bash
python scripts/smoke_e2e.py --csv data/sample.csv --seed 1337 --output-dir reports/smoke-e2e
```

Generated artifacts (plots, metrics) are written to the specified `--output-dir`.

### Hypothesis Configuration

Property-based tests use Hypothesis with these default settings:
- `max_examples=100` for standard tests
- `max_examples=50` for expensive tests
- `deadline=None` to avoid flaky timeout failures

Adjust settings in test decorators:
```python
@settings(max_examples=200, deadline=5000)
@given(data=st.integers())
def test_something(data):
    ...
```

Run with Hypothesis statistics:
```bash
pytest tests/property/ --hypothesis-show-statistics
```

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions.

### CI Workflow

The testing automation is split across two workflows:

- `.github/workflows/tests.yml` (per push / PR)
  1. **Unit & Integration Tests**: Executed with branch coverage gates (line ≥ 90%, branch ≥ 90%).
  2. **Property-Based Tests**: Run with Hypothesis statistics enabled.
  3. **Fuzz Tests**: Replay deterministic fuzz corpora.
  4. **Contract Tests**: Enforce DTO JSON Schema/OpenAPI stability.
  5. **Data Quality Gates**: Fail fast if golden/reference datasets degrade.
  6. **Security Guardrails**: Ensure audit logging keeps secrets out of logs.
  7. **Coverage Report**: Generated and uploaded to CI artifacts, plus Codecov uploads and GitHub step summaries.
  8. **UI Smoke & Accessibility**: Playwright smoke plus aXe scans for critical dashboards.
  9. **Flaky Quarantine**: A dedicated job re-runs `@pytest.mark.flaky` tests with retries and publishes a manifest without blocking the main pipeline.
- **`heavy-math.yml` (per PR, nightly)**
  1. Executes the heavy-math suites defined in `configs/quality/heavy_math_jobs.yaml`.
  2. Enforces CPU/memory quotas via workflow dispatch inputs.
  3. Runs portability checks marked `arm` to assert CPU/GPU/ARM parity.
  4. Blocks merge if any heavy-math job or portability test fails.

- `.github/workflows/mutation-tests.yml` (weekly + manual)
  1. Runs targeted mutation testing against `core/`, `backtest/`, and `execution/`.
  2. Publishes mutation reports and caches as CI artifacts for triage.
  3. Reuses the pytest coverage data to short-circuit equivalent mutants.

- `.github/workflows/smoke-e2e.yml` (nightly + manual)
  1. **E2E Pipeline**: Executes `python scripts/smoke_e2e.py` against `data/sample.csv`.
  2. **Artifact Upload**: Persists generated signals and reports for inspection.

### Reproducing the Tests workflow locally with `act`

Use the repository-scoped `.actrc` to run the GitHub Actions test workflow in Docker containers that mirror CI. The configuration pins `ubuntu-latest` to `ghcr.io/catthehacker/ubuntu:act-latest`, applies the `linux/amd64` container architecture, and defines a `tests` profile wired to `.github/workflows/tests.yml`. Supporting environment and secret bootstrap files live in `.github/act/tests.env` and `.github/act/tests.secrets` respectively (the latter provides a placeholder `GITHUB_TOKEN`).

Run the full workflow matrix exactly as CI does:

```bash
act --profile tests
```

Target an individual job within the workflow by combining `--profile` with `--job` (the job ID is `tests`):

```bash
act --profile tests --job tests
```

Limit execution to a specific matrix entry by supplying one or more `--matrix` selectors. The syntax is `<key>:<value>` and can be repeated to pin multiple axes:

```bash
act --profile tests --job tests --matrix python-version:3.12
```

All commands automatically consume the env/secret files referenced by the profile, so no manual exports are required. Update the placeholder secrets if you need authenticated calls during local runs.

### Coverage Enforcement

The CI enforces both line and branch coverage thresholds using `pytest-cov`:

```bash
pytest --cov=core --cov=backtest --cov=execution --cov=analytics \
       --cov-branch \
       --cov-report=term-missing \
       --cov-fail-under=90
```

To test locally with the same threshold:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics --cov-branch --cov-fail-under=90
```

Additionally, the CI workflow parses `coverage.xml` and fails early if total branch coverage drops below 90%.

### Flaky Test Quarantine

Mark unstable scenarios with `@pytest.mark.flaky`. The quarantine job in CI and local helpers allow you to exercise them without
penalising the main coverage gates:

```bash
pytest tests/ -m flaky --reruns=2 --reruns-delay=2 \
  --flaky-report=reports/flaky-tests.json
```

The `--flaky-report` flag writes a JSON manifest describing reruns, outcomes, and marker metadata, making it easy to monitor
behaviour over time.

### Mutation Testing

Mutation analysis is enforced for the core trading engine modules using [`mutmut`](https://mutmut.readthedocs.io/). The default configuration lives in `pyproject.toml` and targets `core/`, `backtest/`, and `execution/` with the unit, integration, and property suites as the runner.

Run the mutation suite locally (requires the `dev` extras or `requirements-dev.txt`):

```bash
mutmut run --use-coverage
mutmut results
```

The command caches results under `.mutmut-cache/` and reuses coverage data to accelerate re-runs. Use `mutmut show <mutation-id>` to inspect surviving mutants in detail.

## Writing Tests

### Unit Test Guidelines

- Test one function or class per test file
- Use descriptive test names: `test_<what>_<condition>_<expected_result>`
- Keep tests independent and isolated
- Mock external dependencies
- Test edge cases and error conditions

Example:
```python
def test_position_sizing_rejects_invalid_price() -> None:
    """Negative or zero price should raise ValueError."""
    with pytest.raises(ValueError, match="price must be positive"):
        position_sizing(1000.0, 0.5, 0.0)
```

### Property-Based Test Guidelines

- Test invariants that should always hold
- Use Hypothesis to generate test cases
- Focus on properties, not specific values
- Handle edge cases automatically

Example:
```python
from hypothesis import given, strategies as st

@given(
    balance=st.floats(min_value=1.0, max_value=1_000_000.0),
    risk=st.floats(min_value=0.0, max_value=1.0),
)
def test_position_size_never_exceeds_balance(balance: float, risk: float) -> None:
    size = position_sizing(balance, risk, 100.0)
    assert size * 100.0 <= balance * risk * 1.01
```

### Integration Test Guidelines

- Test complete workflows from start to finish
- Use realistic data
- Verify end-to-end behavior
- Test error recovery and resilience

Example:
```python
def test_csv_ingestion_to_strategy_evaluation(tmp_path) -> None:
    """Test complete flow from CSV to strategy evaluation."""
    # Create CSV -> Ingest -> Convert -> Evaluate
    # Assert final result is valid
```

### Fuzz Test Guidelines

- Test with malformed, corrupted, and edge-case data
- Verify graceful error handling
- Ensure no crashes or data corruption
- Use Hypothesis for property-based fuzzing

Example:
```python
@given(rows=st.lists(st.fixed_dictionaries({
    "price": st.one_of(st.floats(), st.just("invalid"))
})))
def test_csv_handles_malformed_data(rows) -> None:
    # Should not crash with invalid data
```

## Test Coverage Best Practices

1. **Test Happy Paths**: Normal, expected usage
2. **Test Edge Cases**: Empty data, single element, extreme values
3. **Test Error Conditions**: Invalid inputs, missing data, corruption
4. **Test Boundaries**: Min/max values, type boundaries
5. **Test Invariants**: Properties that must always hold

## Performance Testing

For large datasets or long-running operations:

```bash
# Mark tests as slow
@pytest.mark.slow
def test_large_dataset():
    ...

# Run only fast tests (skip slow)
pytest tests/ -m "not slow"

# Run only slow tests
pytest tests/ -m "slow"
```

## Debugging Failed Tests

Show full output:
```bash
pytest tests/ -v
```

Show full tracebacks:
```bash
pytest tests/ --tb=long
```

Drop into debugger on failure:
```bash
pytest tests/ --pdb
```

Show print statements:
```bash
pytest tests/ -s
```

## Common Issues

### Import Errors

Ensure the project root is in `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/TradePulse:$PYTHONPATH
```

Or install in development mode:
```bash
pip install -e .
```

### Hypothesis Health Checks

If using fixtures with `@given`, suppress health checks:
```python
from hypothesis import HealthCheck, settings

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.integers())
def test_with_fixture(data, tmp_path):
    ...
```

### Flaky Tests

- Increase Hypothesis deadline: `@settings(deadline=None)`
- Use fixed random seeds: `np.random.seed(42)`
- Avoid time-based assertions

## Contact & Support

For questions about testing:
- Review existing tests for examples
- Check GitHub Issues for known testing issues
- Consult the pytest and Hypothesis documentation

## References

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
