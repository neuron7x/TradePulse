# TradePulse Patch v4-mini

Готово до GitHub: лінт, тести, CI.

## Testing

Comprehensive test suite with 56%+ coverage (target: 98%):

- **Unit Tests**: Individual module tests
- **Integration Tests**: End-to-end workflow tests
- **Property-Based Tests**: Hypothesis-driven invariant testing
- **Fuzz Tests**: Malformed data and edge case handling
- **Performance Tests**: Large dataset stress testing

### Quick Start

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-report=html

# Skip slow tests during development
pytest tests/ -m "not slow"

# Run only property-based tests
pytest tests/property/

# Run only integration tests
pytest tests/integration/
```

See [TESTING.md](TESTING.md) for comprehensive testing documentation.
