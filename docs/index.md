# TradePulse Documentation Portal

Welcome to the TradePulse knowledge base. This site organises the core
concepts, APIs, and operational guides required to build, validate, and run
quantitative trading strategies with the platform. Each section links to a
focused document so contributors can quickly find the details they need.

---

## Architecture Overview

- **High-level system design** – see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
  for component boundaries, data flow, and deployment topologies.
- **FPM-A phase/regime model** – the methodology that powers TradePulse’s
  adaptive agents is described in [`docs/FPM-A.md`](FPM-A.md), including phase
  catalogues, orchestration contracts, and reproducible benchmarks.
- **Monitoring & operations** – instrumentation, alerting, and dashboard
  recipes live in [`docs/monitoring.md`](monitoring.md).
- **Reliability & SRE** – SLA commitments, SLOs, error budgets, and escalation
  policy are outlined in [`docs/reliability.md`](reliability.md).

---

## Core Capabilities

| Topic | What you will learn | Primary References |
| ----- | ------------------- | ------------------ |
| Indicators | Kuramoto phase sync, entropy, Hurst exponent, Ricci curvature, and how to assemble feature blocks. | [`docs/indicators.md`](indicators.md), [`core/indicators`](../core/indicators) |
| Agent System | Strategy lifecycle, mutation/repair flow, instability detection, and multi-armed bandits. | [`docs/agent.md`](agent.md), [`core/agent`](../core/agent) |
| Backtesting | Deterministic walk-forward engine, signal requirements, and diagnostics. | [`docs/backtest.md`](backtest.md), [`backtest/engine.py`](../backtest/engine.py) |
| Execution & Risk | Order sizing utilities, portfolio heat checks, and integration touchpoints. | [`docs/execution.md`](execution.md), [`execution`](../execution) |
| CLI & Interfaces | Automations for analysis/backtest/live commands and scripting entrypoints. | [`interfaces/cli.py`](../interfaces/cli.py) |

---

## Getting Started

1. **Follow the [Quick Start Guide](quickstart.md)** to install dependencies,
   run the CLI, and execute your first analysis.
2. **Study the [Developer Scenarios](scenarios.md)** for task-oriented
   tutorials that cover indicator creation, backtests, live ingestion, and
   strategy optimisation.
3. **Explore [Extending TradePulse](extending.md)** when you are ready to add
   custom indicators, strategies, or integrations.

---

## Contribution Workflow

- Adhere to the testing instructions in [`TESTING.md`](../TESTING.md) and the
  quality checklist documented in [`reports/release_readiness.md`](../reports/release_readiness.md).
- Keep documentation in sync with the public APIs exposed by modules such as
  [`core/agent/strategy.py`](../core/agent/strategy.py) and
  [`execution/order.py`](../execution/order.py).
- When adding major features, update this index so the navigation reflects the
  newest capabilities.
- Apply the practices in the [Technical Debt Management Playbook](technical-debt.md)
  to sustain healthy engineering velocity while avoiding unplanned outages.
- Use the [Quality DoR/DoD checklist](quality-dor-dod.md) to enforce clear entry
  and exit criteria before pulling or closing quality-focused work.

---

**Last updated:** 2025-10-08
