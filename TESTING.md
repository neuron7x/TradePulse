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

## Test Structure

```
tests/
├── unit/                    # Module-level tests (core.*, backtest.*, execution.*, ...)
├── integration/             # Workflow-level tests (pipelines, backtests, protocol adapters)
├── property/                # Property-based suites powered by Hypothesis
├── fuzz/                    # Fuzz harnesses targeting ingestion and message handling
├── e2e/                     # Pytest smoke scenarios that mimic end-user flows
└── fixtures/                # Shared fixtures, builders, and reusable data

scripts/
└── smoke_e2e.py             # CLI entrypoint used by the E2E workflow
```

## Coverage Requirements

**Target Coverage: 98%** *(current CI gate: 90% while Kuramoto/Ricci suites stabilize)*

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

Run with coverage report:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-report=term-missing
```

Generate HTML coverage report:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-report=html
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

**E2E smoke tests (pytest harness):**
```bash
pytest tests/e2e/
```

**Cross-architecture indicator parity (CPU/GPU/ARM simulacrum):**
```bash
pytest -m arm tests/performance/test_indicator_portability.py
```

**Heavy-math validation gate:**
```bash
pytest -m heavy_math tests/unit/config/test_heavy_math_jobs.py
```

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
  1. **Unit & Integration Tests**: Executed with coverage gates.
  2. **Property-Based Tests**: Run with Hypothesis statistics enabled.
  3. **Fuzz Tests**: Replay deterministic fuzz corpora.
  4. **Coverage Report**: Generated and uploaded to CI artifacts.
  5. **Coverage Threshold**: Build fails if coverage `< 90%`.
- **`heavy-math.yml` (per PR, nightly)**
  1. Executes the heavy-math suites defined in `configs/quality/heavy_math_jobs.yaml`.
  2. Enforces CPU/memory quotas via workflow dispatch inputs.
  3. Runs portability checks marked `arm` to assert CPU/GPU/ARM parity.
  4. Blocks merge if any heavy-math job or portability test fails.

- `.github/workflows/smoke-e2e.yml` (nightly + manual)
  1. **E2E Pipeline**: Executes `python scripts/smoke_e2e.py` against `data/sample.csv`.
  2. **Artifact Upload**: Persists generated signals and reports for inspection.

### Coverage Enforcement

The CI enforces coverage thresholds using `pytest-cov`:

```bash
pytest --cov=core --cov=backtest --cov=execution \
       --cov-report=term-missing \
       --cov-fail-under=90
```

To test locally with the same threshold:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-fail-under=90
```

### Mutation Testing

Critical execution and indicator modules are validated with mutation testing.
CI invokes [`mutmut`](https://mutmut.readthedocs.io/en/latest/) against
`core/indicators`, `backtest/engine.py`, and the execution risk/OMS stack to
ensure the regression suite catches behavioural drifts. To reproduce locally:

```bash
# Prime coverage data so mutmut can focus on exercised lines
coverage run -m pytest tests/unit/ tests/integration/

# Execute the configured mutation campaign
mutmut run --use-coverage --no-progress

# Inspect remaining survivors (if any)
mutmut results
```

Mutation runs are intentionally scoped to high-signal modules to keep turnaround
times reasonable while guarding the risk engine and indicator analytics.

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
