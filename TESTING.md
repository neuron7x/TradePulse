# Testing Guide for TradePulse

This document describes the testing strategy, coverage requirements, and instructions for running tests in the TradePulse project.

## Overview

TradePulse employs a comprehensive testing strategy that includes:

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test complete workflows and pipeline chains
- **Property-Based Tests**: Test invariants and properties using Hypothesis
- **Fuzz Tests**: Test robustness with malformed and edge-case data
- **Performance Tests**: Test behavior with large datasets (planned)

### Dependency Matrix

| Layer              | Required Packages                                               | Notes |
| ------------------ | --------------------------------------------------------------- | ----- |
| Runtime            | `numpy`, `pandas`, `scipy`, `PyYAML`                             | PyYAML is mandatory for YAML-driven CLI workflows and configuration fixtures. |
| Test-only          | `pytest`, `pytest-cov`, `hypothesis`, `faker`, `pytest-benchmark` | Install via `requirements-dev.txt`; Hypothesis is treated as a first-class dependency and should be present in local and CI runs. |
| Tooling & linting  | `ruff`, `mypy`, `pre-commit`                                     | Executed through `make lint` and enforced in CI. |

> **Tip:** When creating ephemeral CI environments, install both `requirements.txt` and `requirements-dev.txt` to avoid the historical `ModuleNotFoundError: No module named 'yaml'` failure.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual modules
│   ├── test_agents.py
│   ├── test_data_ingestion.py
│   ├── test_execution.py
│   └── ...
├── integration/             # Integration tests for complete workflows
│   ├── test_pipeline.py
│   ├── test_backtest.py
│   └── test_extended_pipeline.py
├── property/                # Property-based tests
│   ├── test_backtest_properties.py
│   ├── test_execution_properties.py
│   ├── test_strategy_properties.py
│   └── test_invariants.py
├── fuzz/                    # Fuzz tests for robustness
│   └── test_ingestion_fuzz.py
└── fixtures/                # Shared test fixtures and data
    └── conftest.py
```

## Coverage Requirements

**Target Coverage: 98%**

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

Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
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

### Integration Workflow Quickstart

Execute the following sequence on every feature branch before opening a pull request:

1. `pytest -q` – fast validation of unit, integration, property, and fuzz suites.
2. `pytest --cov=core --cov=backtest --cov=execution --cov-report=xml --cov-fail-under=98` – enforces the published coverage threshold.
3. `python -m interfaces.cli backtest configs/backtests/sample.yaml` – exercises YAML parsing and CLI wiring.
4. `make lint` – runs ruff, mypy, gofmt, eslint, and prettier (where applicable).
5. `make docs` – ensures MkDocs builds with updated navigation and references.

These steps mirror the [Quality Assurance Playbook](docs/quality-assurance.md) and help catch dependency regressions early.

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

See `.github/workflows/tests.yml`:

1. **Unit & Integration Tests**: Run with coverage
2. **Property-Based Tests**: Run with Hypothesis statistics
3. **Coverage Report**: Generate and upload to CI artifacts
4. **Coverage Threshold**: Fail if coverage < 98%

### Coverage Enforcement

The CI enforces coverage thresholds using `pytest-cov`:

```bash
pytest --cov=core --cov=backtest --cov=execution \
       --cov-report=term-missing \
       --cov-fail-under=98
```

To test locally with the same threshold:
```bash
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-fail-under=98
```

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
