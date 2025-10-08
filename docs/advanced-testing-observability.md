# Advanced Testing & Observability Features

This document describes the enhanced testing and observability features added to TradePulse.

## 🧪 Test-Driven Evolution (TDE)

### Test Coverage

TradePulse now has **314 comprehensive tests** covering:

- **Unit Tests** (100+): Individual module and function tests
- **Integration Tests** (20+): End-to-end workflow tests
- **Property-Based Tests** (40+): Hypothesis-driven invariant testing
- **Async Tests** (8): Asynchronous data processing tests
- **Fuzz Tests** (15+): Robustness testing with malformed inputs
- **Performance Tests** (10+): Stress testing with large datasets
- **Chaos Tests** (22): Fault-injection and edge case testing

### Coverage by Module

| Module | Coverage | Tests |
|--------|----------|-------|
| `core/utils/security.py` | 100% | 26 |
| `core/utils/logging.py` | 100% | 19 |
| `core/utils/schemas.py` | 97% | 25 |
| `core/utils/metrics.py` | 93% | 28 |

### Mutation Testing

Mutation testing with `mutmut` is configured to verify test effectiveness:

```bash
# Run mutation tests
mutmut run

# Check results
mutmut results

# Show surviving mutants
mutmut show
```

Configuration in `setup.cfg`:
```ini
[mutmut]
paths_to_mutate = core/,backtest/,execution/
exclude = core/utils/__init__.py,conftest.py
runner = pytest -x -q tests/
```

### Chaos/Fault-Injection Tests

Chaos tests verify system behavior under adverse conditions:

```bash
# Run chaos tests
pytest tests/chaos/ -v

# Run specific chaos test category
pytest tests/chaos/test_data_chaos.py -v
```

Chaos tests cover:
- Network failures (ConnectionError, TimeoutError, OSError)
- Data integrity under extreme values (inf, -inf, nan, very large/small numbers)
- Edge cases and boundary conditions
- Special float values and precision

## 🔍 Runtime Validation with Pydantic

Pydantic models provide runtime type checking and validation:

```python
from core.utils.validation import validate_ticker, TickerModel

# Validate ticker data
data = {"ts": 1609459200.0, "price": 50000.0, "volume": 100.0}
validated = validate_ticker(data)

# Use Pydantic model directly
ticker = TickerModel(ts=1609459200.0, price=50000.0, volume=100.0)
```

### Available Validation Models

- **TickerModel**: Market data ticks with positive price validation
- **FeatureResultModel**: Feature/indicator results with metadata
- **BacktestResultModel**: Backtest results with non-positive max drawdown
- **OrderModel**: Order specifications with side and type validation
- **StrategyConfigModel**: Strategy configurations with parameter limits

### Validation Features

- **Strict typing**: Enforces correct data types
- **Range validation**: Ensures values are within reasonable bounds
- **Pattern matching**: Validates string patterns (e.g., order side must be "buy" or "sell")
- **Custom validators**: Complex validation logic for business rules
- **Frozen models**: Immutable data structures where appropriate

## 📊 Distributed Tracing with OpenTelemetry

OpenTelemetry integration provides end-to-end observability:

```python
from core.utils.tracing import trace_operation, trace_function, configure_tracing

# Configure tracing
configure_tracing(service_name="tradepulse", enabled=True)

# Trace a function
@trace_function("compute_indicator")
def compute_rsi(data):
    return rsi_values

# Trace an operation
with trace_operation("backtest", {"strategy": "momentum"}):
    results = run_backtest(data, strategy)
```

### Tracing Features

- **Automatic span creation**: Track execution flow
- **Attribute attachment**: Add context to spans
- **Function/method decorators**: Easy instrumentation
- **Context propagation**: Trace across service boundaries
- **Integration ready**: Works with Jaeger, Zipkin, and other backends

### Tracing Backends

Configure exporters for your preferred backend:

```python
# Console exporter (default, for development)
# Logs traces to console

# OTLP exporter (for production)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Jaeger exporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Zipkin exporter
from opentelemetry.exporter.zipkin.json import ZipkinExporter
```

## 🔐 Enhanced Security Scanning

Comprehensive security scanning with automated detection:

```python
from core.utils.security import SecretDetector, check_for_hardcoded_secrets

# Scan for hardcoded secrets
check_for_hardcoded_secrets(".")

# Custom secret detection
detector = SecretDetector(custom_patterns={
    "custom_key": re.compile(r"custom_pattern")
})
findings = detector.scan_directory("src/")
```

### Security Features

- **Secret pattern detection**: API keys, passwords, tokens, private keys
- **Multi-service support**: AWS, GitHub, Slack, Stripe, etc.
- **Directory scanning**: Recursive search with file filtering
- **Ignore patterns**: Skip test files, docs, and specific directories
- **Safe reporting**: Masks secrets in output

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: secret-scan
      name: Scan for hardcoded secrets
      entry: python -c "from core.utils.security import check_for_hardcoded_secrets; import sys; sys.exit(check_for_hardcoded_secrets('.'))"
      language: python
      pass_filenames: false
```

## 📋 JSON Schema Generation

Auto-generate JSON schemas for all public APIs:

```python
from core.utils.schemas import generate_all_schemas, save_schemas

# Generate schemas
schemas = generate_all_schemas()

# Save to files
save_schemas("docs/schemas")

# Validate data
from core.utils.schemas import validate_against_schema
is_valid = validate_against_schema(data, schemas["FeatureResult"])
```

### Schema Features

- **Automatic generation**: From Python dataclasses
- **OpenAPI compatible**: Use in API documentation
- **Type inference**: Handles complex types (List, Dict, Optional)
- **Validation**: Basic schema validation included
- **Documentation**: Generates description from docstrings

## 📈 Metrics Collection

Prometheus metrics for all entrypoints:

```python
from core.utils.metrics import get_metrics_collector

collector = get_metrics_collector()

# Measure feature transformation
with collector.measure_feature_transform("RSI", "indicator"):
    result = compute_rsi(data)

# Measure backtest
with collector.measure_backtest("momentum"):
    backtest_results = run_backtest(data)

# Record values
collector.record_feature_value("RSI", 65.5)
collector.record_order_placed("buy", "market", "success")
```

### Available Metrics

- **Feature transformations**: Duration, count, values
- **Backtests**: Duration, count, status
- **Data ingestion**: Ticks processed, errors
- **Order execution**: Orders placed, execution time, status

### Metrics Server

Start metrics HTTP server:

```python
from core.utils.metrics import start_metrics_server

# Start on default port (8000)
start_metrics_server()

# Custom port
start_metrics_server(port=9090)
```

Access metrics at `http://localhost:8000/metrics`

## 🔬 Structured Logging

JSON-formatted logs with correlation IDs:

```python
from core.utils.logging import get_logger, configure_logging

# Configure logging
configure_logging(level="INFO", use_json=True)

# Get logger
logger = get_logger("mymodule", correlation_id="req-123")

# Log with structured fields
logger.info("Processing order", order_id=456, symbol="BTCUSD")

# Track operations
with logger.operation("compute_features", dataset_size=1000):
    features = compute_all_features(data)
```

### Logging Features

- **JSON formatting**: Structured, parseable logs
- **Correlation IDs**: Track requests across services
- **Operation tracking**: Automatic timing and status
- **Extra fields**: Add custom context to logs
- **Exception handling**: Automatic exception info capture

## 🎯 Best Practices

### Testing

1. **Write tests first**: Follow TDD principles
2. **Property-based testing**: Use Hypothesis for invariant testing
3. **Chaos testing**: Test edge cases and failure scenarios
4. **Mutation testing**: Verify test effectiveness
5. **Integration tests**: Test complete workflows

### Observability

1. **Structured logging**: Always use JSON format in production
2. **Correlation IDs**: Generate and propagate through all operations
3. **Distributed tracing**: Instrument all service boundaries
4. **Metrics collection**: Record all business-critical operations
5. **Error tracking**: Log errors with full context

### Security

1. **No hardcoded secrets**: Use environment variables or secret managers
2. **Regular scanning**: Run security scans in CI/CD
3. **Dependency updates**: Keep dependencies up-to-date
4. **Input validation**: Validate all external inputs with Pydantic
5. **Least privilege**: Use minimal permissions for all operations

### Validation

1. **Runtime validation**: Use Pydantic for all external inputs
2. **Type hints**: Annotate all functions and methods
3. **Schema validation**: Validate against JSON schemas
4. **Boundary checking**: Validate ranges and constraints
5. **Error messages**: Provide clear validation error messages

## 📚 References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [Mutation Testing](https://mutmut.readthedocs.io/)
