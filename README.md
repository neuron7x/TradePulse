# TradePulse

**Quantitative trading research & execution platform built around geometric market intelligence.**

[![Tests Status](https://img.shields.io/github/actions/workflow/status/neuron7x/TradePulse/tests.yml?branch=main&label=tests)](https://github.com/neuron7x/TradePulse/actions/workflows/tests.yml)
[![Security Scan](https://img.shields.io/github/actions/workflow/status/neuron7x/TradePulse/security.yml?branch=main&label=security)](https://github.com/neuron7x/TradePulse/actions/workflows/security.yml)
[![Coverage](https://img.shields.io/codecov/c/github/neuron7x/TradePulse?branch=main&label=coverage)](https://app.codecov.io/gh/neuron7x/TradePulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Async: asyncio](https://img.shields.io/badge/async-asyncio-green.svg)](https://docs.python.org/3/library/asyncio.html)
[![Metrics: Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange.svg)](https://prometheus.io/)

> TradePulse combines multi-scale Kuramoto synchronisation, Ricci curvature analytics, entropy metrics and adaptive agents into a cohesive toolkit for quantitative research teams. It ships with event-driven backtesting, execution engines, GPU accelerators, observability assets, and documentation powered by MkDocs.

---

## 🧭 Contents

- [Overview](#-overview)
- [Platform Capabilities](#-platform-capabilities)
- [Architecture & Components](#-architecture--components)
- [Data & Indicators](#-data--indicators)
- [Workflow Playbooks](#-workflow-playbooks)
- [Installation](#-installation)
- [CLI Reference](#-cli-reference)
- [Observability & Ops](#-observability--ops)
- [Quality & CI](#-quality--ci)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

TradePulse is engineered as a cross-language research platform for professional quants. Python modules implement signal generation, feature engineering, and orchestration; Go services handle latency-sensitive market microstructure analysis. Async pipelines, GPU fallbacks, and strict typing make it possible to move seamlessly between research notebooks and production trading workflows.

Use TradePulse to:

- Build composite market regimes with Kuramoto, Ricci, entropy, and Hurst indicators.
- Run walk-forward simulations that mirror production latency and fee constraints.
- Execute adaptive market-making or directional strategies with built-in risk controls.
- Observe live systems through Prometheus metrics, structured logging, and Grafana dashboards.

---

## 🚀 Platform Capabilities

### Research & Signal Generation
- `core/indicators/` provides Kuramoto, multiscale Kuramoto, Ricci curvature, entropy, and Hurst exponent implementations with caching utilities.
- `core/phase/` fuses indicators into phase transition detectors, while `core/metrics/` codifies evaluation metrics.
- `core/data/` contains adapters for CSV/stream ingestion, async pipelines, validation, and preprocessing helpers (e.g., float32 scaling, chunked transformations).
- GPU acceleration is available through `compute_phase_gpu` when CuPy is installed, enabling high-frequency experiments.

### Portfolio & Execution
- `backtest/` implements the event-driven engine, walk-forward runner, strategy hooks, resampling, and transaction-cost modelling.
- `execution/` houses order models, AMM integration, and risk controls designed for latency-aware execution.
- `interfaces/` exposes CLIs, ingestion layers, and dashboard streaming endpoints; `apps/web/` contains the Next.js monitoring dashboard.

### Operational Excellence
- `observability/` packages Prometheus metrics definitions, alert policies, and Grafana dashboards for monitoring deployments.
- `reports/` aggregates CI outputs (coverage, static analysis, dependency audit) that support release readiness reviews.
- `deploy/` and `docker-compose.yml` encode reproducible environments for local clusters or staging rollouts.

---

## 🏗️ Architecture & Components

| Layer | Path(s) | Responsibilities |
| --- | --- | --- |
| Signal Intelligence | [`core/indicators/`](core/indicators/), [`core/phase/`](core/phase/), [`core/neuro/`](core/neuro/) | Indicator math, composite regime detection, adaptive agent primitives. |
| Data Foundation | [`core/data/`](core/data/), [`data/`](data/), [`sample.csv`](sample.csv) | Data ingestion adapters, preprocessing, canonical datasets for benchmarks. |
| Strategy Lifecycle | [`backtest/`](backtest/), [`core/agent/`](core/agent/), [`core/strategies/`](core/strategies/) | Walk-forward backtests, genetic agent optimisation, reusable strategy templates. |
| Execution & Risk | [`execution/`](execution/), [`interfaces/execution.py`](interfaces/execution.py) | Order routing abstractions, AMM runners, protective risk controls. |
| Interfaces & Apps | [`interfaces/`](interfaces/), [`apps/web/`](apps/web/), [`cli/`](cli/) | CLI entry points, Streamlit dashboards, web applications. |
| Go Microservices | [`markets/`](markets/), [`analytics/`](analytics/) | VPIN calculators, order-book analytics, regime detection engines in Go. |
| Tooling & Docs | [`docs/`](docs/), [`mkdocs.yml`](mkdocs.yml), [`observability/`](observability/), [`buf.yaml`](buf.yaml) | MkDocs site, monitoring assets, protobuf definitions (via Buf), operational guides. |

Refer to [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for in-depth diagrams and design principles.

---

## 📊 Data & Indicators

- **Kuramoto Synchronisation** (`core/indicators/kuramoto.py`, `core/indicators/multiscale_kuramoto.py`) analyses collective price phases across time scales.
- **Ricci Curvature** (`core/indicators/ricci.py`, `core/indicators/temporal_ricci.py`) measures structural stress within price graphs to anticipate regime shifts.
- **Entropy & Delta Entropy** (`core/indicators/entropy.py`) quantify uncertainty dynamics, with configurable binning and float32 support.
- **Hurst Exponent** (`core/indicators/hurst.py`) classifies persistence vs. mean-reversion regimes.
- **Composite Transitions** (`core/phase/detector.py`) merges the above signals into interpretable phase flags and triggers.
- **Agent Toolkit** (`core/agent/`) includes reinforcement and evolutionary components for automated strategy search.
- **Datasets** live under [`data/`](data/) and [`sample.csv`](sample.csv), while [`docs/dataset_catalog.md`](docs/dataset_catalog.md) documents recommended market feeds.

---

## 🔁 Workflow Playbooks

### Quant Research Loop
1. Start in a notebook or script using primitives from `core/indicators` and `core/data.preprocess` to explore candidate alphas.
2. Persist experiments using [`reports/`](reports/) conventions or [`analytics/`](analytics/) services for reproducible datasets.
3. Promote promising features into `core/strategies/` or `core/agent/` for automated tuning.

### Walk-Forward Backtesting
```bash
# Example: evaluate a strategy with realistic transaction costs
python -m interfaces.cli backtest --csv sample.csv \
    --train-window 500 --test-window 100 \
    --fee 0.0005
```
- `backtest/engine.py` orchestrates walk-forward windows, while `backtest/performance.py` reports PnL, drawdowns, and trade stats.
- Extend behaviour via `backtest/strategies/` or by injecting callbacks into `walk_forward`.

### Live Execution Demo
```bash
# Simulate live processing from a CSV feed with GPU acceleration
python -m interfaces.cli live --source csv --path sample.csv --window 200 --gpu
```
- `core/data.ingestion.DataIngestor` streams ticks, `core/phase/detector.py` emits composite signals, and JSON output is ready for downstream execution engines.
- For production integrations, connect interfaces in [`execution/`](execution/) or bridge to Go services under [`markets/`](markets/).

---

## 🧰 Installation

### Python Environment
```bash
git clone https://github.com/neuron7x/TradePulse.git
cd TradePulse
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # runtime stack
pip install -r requirements-dev.txt      # linting, testing, docs
```
- Optional extras: install `cupy` for GPU-accelerated phase computations and `pyyaml` for CLI config overlays.

### Docker Compose
```bash
docker compose up -d        # start ingestion, analytics, and dashboards
docker compose logs -f      # follow service output
docker compose down         # stop the stack
```
See [docs/docker-quickstart.md](docs/docker-quickstart.md) for GPU passthrough, environment variables, and troubleshooting.

### Documentation Site
```bash
pip install -r requirements-dev.txt
mkdocs serve -f mkdocs.yml  # local docs portal at http://127.0.0.1:8000
```
The MkDocs configuration mirrors the structure of [`DOCUMENTATION_SUMMARY.md`](DOCUMENTATION_SUMMARY.md).

---

## 🛎️ CLI Reference

| Command | Description | Key Options |
| --- | --- | --- |
| `python -m interfaces.cli analyze` | Compute Kuramoto, entropy, Ricci, and Hurst metrics from CSV data. | `--csv`, `--window`, `--bins`, `--delta`, `--gpu`, `--config` |
| `python -m interfaces.cli backtest` | Run walk-forward simulations with configurable windows and fees. | `--csv`, `--window`, `--fee`, `--gpu`, `--config` |
| `python -m interfaces.cli live` | Stream signals from CSV (demo) or plug in ingestion adapters. | `--source`, `--path`, `--window`, `--gpu`, `--config` |

Additional entry points live in [`cli/`](cli/) for project automation and [`interfaces/dashboard_streamlit.py`](interfaces/dashboard_streamlit.py) for interactive monitoring.

---

## 📡 Observability & Ops

- Prometheus metrics definitions (`observability/metrics.json`) and alert rules (`observability/alerts.json`).
- Grafana dashboards under `observability/dashboards/` ready for import or automated provisioning.
- Structured logging helpers and correlation IDs described in [docs/monitoring.md](docs/monitoring.md).
- Deployment recipes in [`deploy/`](deploy/) plus infrastructure IaC stubs for clouds and on-prem clusters.

---

## ✅ Quality & CI

| Signal | Description |
| --- | --- |
| [![Tests Status](https://img.shields.io/github/actions/workflow/status/neuron7x/TradePulse/tests.yml?branch=main&label=tests)](https://github.com/neuron7x/TradePulse/actions/workflows/tests.yml) | Matrix running Python `pytest`, `ruff`, `mypy`, and Go `go test ./...` suites.
| [![Coverage](https://img.shields.io/codecov/c/github/neuron7x/TradePulse?branch=main&label=coverage)](https://app.codecov.io/gh/neuron7x/TradePulse) | Codecov ingests `coverage.xml` to expose per-module insights.
| [![Security Scan](https://img.shields.io/github/actions/workflow/status/neuron7x/TradePulse/security.yml?branch=main&label=security)](https://github.com/neuron7x/TradePulse/actions/workflows/security.yml) | Dependency scanning, secret detection, and supply-chain checks.

Review [`reports/`](reports/) for latest pipeline artefacts and [`TESTING_SUMMARY.md`](TESTING_SUMMARY.md) for pass/fail history.

---

## 📖 Documentation

- [Quick Start](docs/quickstart.md) & [Installation](docs/installation.md) cover local setups and prerequisites.
- [Architecture](docs/ARCHITECTURE.md) dives into design decisions, component diagrams, and data flow.
- [Indicators](docs/indicators.md) and [Agent System](docs/agent.md) provide mathematical derivations and usage guidance.
- [Backtesting](docs/backtest.md), [Execution](docs/execution.md), and [Integration API](docs/integration-api.md) explain production deployment pathways.
- Operational manuals: [Monitoring](docs/monitoring.md), [Deployment](docs/deployment.md), [Resilience GameDay](docs/resilience.md), [Security Policy](SECURITY.md).
- Developer resources: [Extending TradePulse](docs/extending.md), [Developer Scenarios](docs/scenarios.md), [Configuration](docs/configuration.md), [Technical Debt Playbook](docs/technical-debt.md).

A curated overview lives in [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) with release highlights summarised in [UPGRADE_SUMMARY.md](UPGRADE_SUMMARY.md).

---

## 🧪 Testing

```bash
pytest -q                               # Python unit + integration suites
go test ./...                           # Go analytics & market microstructure tests
ruff check .                            # Lint for style and static correctness
mypy .                                  # Enforce type guarantees
buf lint && buf generate                # Validate protobuf schemas and regenerate stubs
```
See [TESTING.md](TESTING.md) for fixture design, coverage targets, and CI integration details.

---

## 🤝 Contributing

We welcome contributions from researchers, engineers, and ops teams. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, coding standards, and review expectations. Our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) outlines community behaviour.

---

## 📄 License

TradePulse is released under the MIT License. See [LICENSE](LICENSE) for the full text.

---

## 📞 Contact

- Issues & feature requests: [GitHub Issues](https://github.com/neuron7x/TradePulse/issues)
- Security reports: security@tradepulse.local (PGP details in [SECURITY.md](SECURITY.md))
- General enquiries: maintainers listed in [CODEOWNERS](CODEOWNERS)

---

**TradePulse** – linted, tested, observability-ready, and production focused.
