# Architecture & Repository Design

This document captures the structural blueprint of TradePulse so that every
contributor understands how the repository is organised, how responsibilities
are divided, and which contracts keep the platform cohesive. It is the
companion to the detailed subsystem guides (`docs/agent.md`,
`docs/backtest.md`, `docs/execution.md`, `docs/indicators.md`) and should be
used as the entry point for any architectural decision.

---

## Guiding Principles

1. **Contracts First** – Every cross-module interaction is defined by an
   explicit contract (dataclasses, Protocol Buffers, or typed interfaces). We
   never rely on implicit duck typing for public boundaries.
2. **Deterministic Pipelines** – Backtests, analytics, and production flows must
   be reproducible. File formats, random seeds, and caching strategies are
   documented and versioned.
3. **Fractal Composition** – Indicator pipelines, agents, and execution follow a
   repeating pattern of *feature → block → orchestrator* so that new components
   can be slotted in without bespoke glue code.
4. **Polyglot by Design** – Python orchestrates research workflows, while
   latency-sensitive calculations live in Go. Shared contracts and build tools
   make the hand-off explicit.
5. **Quality Gates Everywhere** – Linters, unit tests, property tests, and
   scenario regressions must pass locally before code is merged. Documentation
   accompanies every new capability.

---

## Repository Topology

```
TradePulse/
├── analytics/           # Go analytics services (regime detection, FPMA)
├── apps/                # User-facing applications (Next.js dashboard)
├── backtest/            # Walk-forward simulation engine
├── configs/             # Configuration presets and YAML manifests
├── core/                # Core trading logic (features, agents, data)
│   ├── agent/           # Strategy lifecycle, optimisation, risk governance
│   ├── data/            # Data ingestion, validation, and caching primitives
│   ├── indicators/      # Feature transformers and composite blocks
│   └── phase/           # Regime detection and market state modelling
├── execution/           # Order routing, venue adapters, and risk utilities
├── interfaces/          # CLI, API surfaces, and automation entry points
├── libs/                # Shared Python utilities (math, yaml shim, telemetry)
├── markets/             # Exchange-specific Go services (VPIN, order book)
├── reports/             # Release readiness and observability reports
├── scripts/             # Developer tooling and operational scripts
└── tests/               # Unit, integration, property, and fuzz suites
```

Each directory is autonomous: a module only imports *downward* (e.g. `core`
may use helpers from `libs`, but `libs` never reaches into `core`). Breaking the
import hierarchy is considered an architectural violation.

---

## Layered Architecture

### 1. Data & Feature Layer
- **Location:** `core/data`, `core/indicators`, selected utilities in `libs/`.
- **Responsibility:** Normalize raw market feeds, compute features, and expose
  deterministic transforms returning `FeatureResult` objects.
- **Key Contracts:**
  - `BaseFeature` (pure transforms) and `FeatureBlock` (feature aggregators).
  - `DataSlice` for time-windowed market inputs.
  - YAML configuration schema (parsed by `yaml.safe_load`).

### 2. Signal & Strategy Layer
- **Location:** `core/agent`, `core/phase`.
- **Responsibility:** Convert feature outputs into tradeable signals, perform
  genetic optimisation, detect instability, and supervise risk envelopes.
- **Key Contracts:**
  - `Strategy.simulate_performance` and `StrategyDiagnostics`.
  - Mutation/repair interfaces for agent genomes.
  - Phase classifiers that emit `PhaseShift` events consumed downstream.

### 3. Execution Layer
- **Location:** `execution`, `markets`, `analytics`.
- **Responsibility:** Route orders, track fills, monitor venues, and run
  low-latency analytics in Go.
- **Key Contracts:**
  - `execution/order.py` order schema and validation utilities.
  - gRPC/Protobuf service definitions (see `buf.yaml`).
  - `markets/*` Go interfaces with explicit adapters for exchange connectivity.

### 4. Interface & Tooling Layer
- **Location:** `interfaces`, `apps`, `scripts`, `reports`.
- **Responsibility:** Provide CLI commands, dashboards, automation scripts, and
  release/observability reports.
- **Key Contracts:**
  - CLI command registry (`interfaces/cli.py`).
  - Next.js API routes aligning with the gRPC gateway.
  - Reporting templates that reference structured JSON/CSV outputs.

---

## Data Flow Overview

1. **Ingestion** – CSV, Parquet, or streaming feeds are parsed into
   `DataSlice` objects (`core/data/loader.py`). Validation guards enforce
   monotonic timestamps and numeric integrity.
2. **Feature Extraction** – `FeatureBlock` instances combine geometric metrics
   (Ricci curvature, Kuramoto order, entropy, etc.). Outputs are cached with a
   deterministic hash so walk-forward splits remain reproducible.
3. **Strategy Simulation** – `Strategy` instances evaluate features against
   trading rules. Diagnostics capture equity curves, drawdowns, and instability
   alerts, even when data windows collapse.
4. **Execution Planning** – The execution module converts signals into orders,
   performs position sizing, and checks heat/VAR limits before orders leave the
   process.
5. **Order Routing** – Go services (`markets/`, `analytics/`) handle heavy
   computations and stream results back via gRPC/gRPC-Web for the dashboard and
   CLI.

All artefacts (signals, orders, diagnostics) are serialised through protobuf
messages defined under `proto/` (generated into `backtest/` and `execution/`).

---

## Configuration & Secrets

- Configuration files live under `configs/` and use the built-in YAML shim
  (`yaml.safe_load`). The schema is versioned; breaking changes require a
  migration note in `CHANGELOG.md`.
- Sensitive values are never committed. Environment variables are loaded via
  `libs/secrets.py` (documented in `SECURITY.md`).
- Defaults are stored in `configs/defaults/*.yaml` and overlaid with
  environment-specific files using a shallow merge helper.

---

## Build & Release Workflow

1. **Python Tooling** – Managed via `pyproject.toml`. Use `poetry` for local
   environment creation and `tox` (coming soon) for multi-version testing.
2. **Go Services** – Modules under `analytics/` and `markets/` share a single
   `go.mod`. Run `make go-test` before committing Go changes.
3. **Front-end** – `apps/web` contains the dashboard. `npm run lint` and
   `npm run test` must pass for UI updates.
4. **CI/CD** – GitHub Actions orchestrate linting, tests, and docker image
   publishing. Pipelines mirror the local make targets (`make lint`,
   `make test`, `make docker-build`).

Release readiness is recorded in `reports/release_readiness.md` and mirrored in
`DOCUMENTATION_SUMMARY.md`.

---

## Testing Strategy

Testing spans four tiers:

| Tier | Location | Purpose |
| ---- | -------- | ------- |
| Unit | `tests/unit` | Validate individual functions/classes and enforce contracts |
| Property | `tests/property` | Hypothesis-based invariants for stochastic algorithms |
| Integration | `tests/integration` | Exercise pipelines end-to-end (CSV → signals → orders) |
| Fuzz/Regression | `tests/fuzz`, `tests/regression` | Guard against malformed input and historical regressions |

Additional expectations:
- Every module exports factories for building fixtures in tests (`tests/fixtures`).
- Slow tests are marked with `@pytest.mark.slow` and excluded from default CI.
- Any bug fix must include a failing test first.

---

## Documentation & Decision Records

- Architectural decisions are logged in `docs/adr/` (use `adr-new` script to
  create entries). The architecture guide summarises stable decisions but ADRs
  capture the rationale and alternatives.
- The `docs/index.md` portal is updated whenever a new capability is added.
- Keep diagrams and tables consistent with the repository structure. When
  directories are added or removed, update both this document and
  `README.md` to reflect the change.

---

## Contribution Checklist (Architecture-Specific)

Before opening a pull request that affects repository design:

1. Update `docs/ARCHITECTURE.md` with any new boundaries or responsibilities.
2. Add or modify ADRs if the change alters an architectural decision.
3. Ensure import layering is preserved (no upward imports).
4. Verify configuration schemas and migrations are documented.
5. Run the full test suite (`pytest -q`) and relevant language-specific tests.

Following this checklist keeps TradePulse aligned with the design principles at
the top of this document and ensures the repository remains approachable to new
contributors and auditors alike.
