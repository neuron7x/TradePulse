# TradePulse

**Advanced algorithmic trading framework powered by geometric market indicators**

[![Tests](https://github.com/neuron7x/TradePulse/workflows/Tests/badge.svg)](https://github.com/neuron7x/TradePulse/actions)
[![Security](https://github.com/neuron7x/TradePulse/workflows/Security%20Scan/badge.svg)](https://github.com/neuron7x/TradePulse/actions)
[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)](./TESTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Async: asyncio](https://img.shields.io/badge/async-asyncio-green.svg)](https://docs.python.org/3/library/asyncio.html)
[![Metrics: Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange.svg)](https://prometheus.io/)

TradePulse is a professional algorithmic trading platform that combines advanced mathematical indicators (Kuramoto synchronization, Ricci curvature, entropy metrics) with modern backtesting and execution capabilities. The framework emphasizes geometric and topological market analysis to detect regime transitions and generate trading signals.

**✨ Key Features:**
- 🔬 **Advanced Indicators**: Kuramoto, Ricci curvature, entropy, Hurst exponent
- ⚡ **Async Support**: Full async/await for data ingestion and processing
- 📊 **Observability**: Structured JSON logging + Prometheus metrics
- 🔒 **Type Safe**: 100% type hints with mypy validation
- 🧪 **Well Tested**: 192 tests with 98% coverage
- 📚 **Documentation**: API docs, tutorials, Jupyter notebooks
- 🔐 **Secure**: Automated security scanning and secret detection

---

## 🚀 Quick Start

### Installation with pip

```bash
# Clone the repository
git clone https://github.com/neuron7x/TradePulse.git
cd TradePulse

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Installation with Docker

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

See [Docker Quick Start Guide](docs/docker-quickstart.md) for detailed instructions.

---

## 📋 Features

### Core Indicators
- **Kuramoto Synchronization**: Phase coherence analysis for market synchronization detection
- **Ricci Curvature**: Geometric curvature analysis on price graphs for regime detection
- **Entropy Metrics**: Shannon entropy and delta entropy for market uncertainty quantification
- **Hurst Exponent**: Long-memory process detection and trend persistence analysis
- **VPIN (Volume-Synchronized Probability of Informed Trading)**: Order flow toxicity metrics

### Trading Capabilities
- **Walk-forward Backtesting**: Realistic simulation with configurable windows
- **Risk Management**: Position sizing, stop-loss, take-profit automation
- **Multi-strategy Support**: Genetic algorithm-driven strategy optimization
- **Async Data Ingestion**: Full async/await support for CSV and streaming data
- **Real-time Execution**: Live trading interface with multiple data sources

### Observability & Operations
- **Structured JSON Logging**: Correlation IDs, operation tracking, hierarchical logging
- **Prometheus Metrics**: Complete instrumentation of features, backtests, data pipelines
- **JSON Schemas**: Auto-generated schemas for all public payloads (OpenAPI compatible)
- **Security Scanning**: Automated secret detection and dependency vulnerability checks
- **Type Safety**: 100% type hints with strict mypy validation (0 errors)
- **Real-time Execution**: Live trading interface with multiple data sources

### Architecture
- **Contracts-first Design**: Protocol Buffers for all data interfaces
- **Fractal Modular Architecture (FPM-A)**: Clean separation of concerns
- **Microservices Ready**: Go engines for performance-critical components
- **Python Execution Loop**: Flexible strategy development and backtesting

---

## 📖 Documentation

### Getting Started
- [Quick Start Guide](docs/quickstart.md) - Get up and running in 5 minutes
- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and principles
- [FAQ](docs/faq.md) - Frequently asked questions
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Core Documentation
- [Indicators](docs/indicators.md) - Mathematical indicators and their usage
- [Backtesting](docs/backtest.md) - Walk-forward simulation and testing
- [Execution](docs/execution.md) - Order execution and risk management
- [Agent System](docs/agent.md) - Genetic algorithm strategy optimization

### Developer Guides
- [Contributing](CONTRIBUTING.md) - Contribution guidelines and workflow
- [Testing Guide](TESTING.md) - Comprehensive testing documentation
- [Extending TradePulse](docs/extending.md) - Adding new indicators and strategies
- [Integration API](docs/integration-api.md) - API reference and integration patterns
- [Developer Scenarios](docs/scenarios.md) - Common development tasks

### Operations
- [Security Policy](SECURITY.md) - Security guidelines and vulnerability reporting
- [Monitoring Guide](docs/monitoring.md) - Metrics, logging, and alerting
- [Deployment](docs/deployment.md) - Production deployment guide

---

## 🧪 Testing

Comprehensive test suite with **98% coverage**:

- **Unit Tests**: Individual module tests (100+ tests)
- **Integration Tests**: End-to-end workflow tests (20+ tests)
- **Property-Based Tests**: Hypothesis-driven invariant testing (40+ tests)
- **Async Tests**: Asynchronous data processing tests (10+ tests)
- **Fuzz Tests**: Robustness testing with malformed inputs (15+ tests)
- **Performance Tests**: Stress testing with large datasets (10+ tests)

**Total: 192 tests, all passing**
- **Property-Based Tests**: Hypothesis-driven invariant testing
- **Fuzz Tests**: Malformed data and edge case handling
- **Performance Tests**: Large dataset stress testing

### Quick Test Commands

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

---

## 📊 Architecture Overview

```
TradePulse/
├── core/               # Core trading logic
│   ├── indicators/     # Mathematical indicators
│   ├── agent/          # Strategy optimization
│   ├── data/           # Data ingestion and preprocessing
│   └── phase/          # Market regime detection
├── backtest/           # Backtesting engine
├── execution/          # Order execution and risk management
├── interfaces/         # CLI and API interfaces
├── markets/            # Market-specific engines (Go)
│   ├── vpin/           # VPIN calculator
│   └── orderbook/      # Order book analyzer
├── analytics/          # Analytics engines (Go)
│   ├── fpma/           # FPM-A complexity analyzer
│   └── regime/         # Regime detection service
├── apps/               # Web applications
│   └── web/            # Next.js dashboard
└── docs/               # Documentation
```

---

## 🎯 Usage Examples

### Analyze Market Data

```bash
# Analyze CSV data
python -m interfaces.cli analyze --csv sample.csv --window 200

# Output includes Kuramoto order, entropy, Ricci curvature, Hurst exponent
```

### Run Backtest

```bash
# Walk-forward backtest with custom strategy
python -m interfaces.cli backtest --csv sample.csv \
    --train-window 500 --test-window 100 \
    --initial-capital 10000
```

### Live Trading (Demo)

```bash
# Simulate live trading from CSV
python -m interfaces.cli live --source csv --path sample.csv --window 200
```

See [Usage Examples](docs/examples/) for more detailed examples.

---

## 🔒 Security

TradePulse implements comprehensive security measures with professional CI/CD practices:

### Automated Security Scanning
- **CodeQL Analysis**: Advanced static analysis for Python security vulnerabilities
- **Bandit**: Python security linter checking for common security issues
- **Secret Scanning**: Multi-tool approach (custom scanner, TruffleHog, Gitleaks)
- **Dependency Scanning**: pip-audit with custom vulnerability filtering
- **Weekly Automated Scans**: Scheduled security audits every Monday

### Security Policies
- **Coordinated Disclosure**: Responsible vulnerability reporting process
- **Code Owner Review**: Required approvals for security-sensitive changes
- **Dependabot**: Automated weekly dependency updates
- **Security Artifacts**: All scan results archived for audit trail (90-day retention)

Please see [SECURITY.md](SECURITY.md) for:
- Vulnerability disclosure process and timelines
- Security best practices for contributors
- Dependency management guidelines
- Security tooling and scanning details
- Severity classification and response SLAs

**Report Security Issues**: security@tradepulse.local or via [GitHub Security Advisories](https://github.com/neuron7x/TradePulse/security/advisories/new)

---

## 📈 Monitoring

The framework includes built-in monitoring capabilities:
- Prometheus metrics export
- Structured logging
- Alert definitions for critical events
- Grafana dashboard templates

See [Monitoring Guide](docs/monitoring.md) for setup and configuration.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow
- Code standards
- PR and issue templates
- Review checklists

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kuramoto Model**: Y. Kuramoto for synchronization theory
- **Ricci Curvature**: Geometric approaches to network analysis
- **VPIN**: Easley, López de Prado, and O'Hara for order flow toxicity metrics

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/neuron7x/TradePulse/issues)
- **Security**: security@tradepulse.local
- **General**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

**TradePulse** - Готово до GitHub: лінт, тести, CI | Ready for production: linted, tested, CI/CD
