# TradePulse Roadmap

This roadmap captures the major platform milestones through **Q4 2025**. Each
milestone is paired with measurable CI health metrics so that progress is
observable and gating remains objective. Targets are reviewed at the end of
every quarter and adjusted only through RFCs.

## 2024

### Q3 2024 — Observability Foundations
- **Goals**
  - Ship golden signal dashboards (RPS, latency p95/p99, error rate, resource
    saturation) with actionable alerting and runbooks.
  - Land automated HTML artifacts for research jobs (backtest/train/publish)
    so every CI run exposes metrics and plots.
  - Harden ingestion/materialisation with chaos tests that simulate network
    drops, delayed writes, and retry logic for transient errors.
- **CI Metrics**
  - Unit + integration coverage ≥ **92%**; property-based coverage report
    published per build.
  - E2E pass rate ≥ **98%** over rolling 14-day window.
  - Mean Time To Recovery (MTTR) for red pipelines < **1 hour** with
    documented remediation steps.

### Q4 2024 — Resilience & Governance
- **Goals**
  - Extend chaos suites to cover feature-store materialisation and async ingest
    batch processing with automated retry verification.
  - Enforce SBOM license validation in CI and publish third-party notices for
    every release artifact.
  - Complete OIDC migration for workflows (no long-lived PAT usage) and enable
    signed release attestations.
- **CI Metrics**
  - Coverage floor raised to **94%** with zero missing HTML coverage uploads.
  - Nightly chaos suite success rate ≥ **95%**.
  - MTTR < **45 minutes** for CI regressions affecting protected branches.

## 2025

### Q1 2025 — Adaptive Research Tooling
- **Goals**
  - Parameter sweep orchestration (“train” alias) emits optimisation trails,
    Pareto front plots, and comparison tables for strategies.
  - Backtest replay harness incorporates production snapshots and equity curve
    drift detection.
  - Establish compliance guardrails: NOTICE file, third-party matrix, and SBOM
    license diffs reviewed automatically.
- **CI Metrics**
  - Research artifact generation success ≥ **99%** with report validation.
  - Backtest/e2e parity drift alarms investigated < **24 hours**.
  - MTTR < **30 minutes** for research job failures on `main`.

### Q2 2025 — Production Scale-Out
- **Goals**
  - Add horizontal scaling benchmarks and saturation modelling for execution
    services.
  - Expand chaos testing to multi-region failover, verifying checkpoint replay
    and materialisation recovery.
  - Integrate service health scoring into deploy gates (latency, error rate,
    saturation) with automated rollback triggers.
- **CI Metrics**
  - Coverage ≥ **95%** with mutation testing parity on core modules.
  - Golden signal alert acknowledgement time < **10 minutes** on average.
  - MTTR < **20 minutes** for release-blocking incidents.

### Q3 2025 — Intelligent Operations
- **Goals**
  - Adaptive alert routing based on blast radius and team ownership with
    embedded runbooks.
  - Automated remediation workflows for common ingest/materialisation faults
    triggered from CI artifacts.
  - Quarterly SBOM + license attestation snapshot published to the trust portal.
- **CI Metrics**
  - E2E pass rate ≥ **99%** with flaky test budget < **0.5%**.
  - Chaos experiment coverage ≥ **90%** of critical data paths.
  - MTTR < **15 minutes** for CI or release blocking alarms.

### Q4 2025 — Enterprise Readiness
- **Goals**
  - Achieve SOC 2-aligned release evidence: signed provenance, NOTICE/third
    party attestations, reproducible research artifacts.
  - Real-time guardrails on data pipelines enforce SLA compliance and embargo
    policies before publish.
  - Unified analytics portal surfaces pipeline health, licensing posture, and
    deployment readiness for compliance review.
- **CI Metrics**
  - Test coverage sustained ≥ **96%**; mutation survivals < **2%**.
  - p99 pipeline duration tracked with upper bound **< 25 minutes**.
  - MTTR < **10 minutes** with automated rollback success rate ≥ **95%**.

---

Progress is tracked publicly in this repository. Each milestone entry links to
supporting documentation (runbooks, dashboards, postmortems) as features land.
