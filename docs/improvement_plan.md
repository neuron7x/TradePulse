# TradePulse Capability Uplift Program (Target 9–10/10)

This program plan operationalises every capability facet into scheduled workstreams,
clear ownership, measurable outcomes, and cadence for review. Use it as the
single source of truth for orchestrating execution.

---

## 1. Program Overview

| Phase | Duration | Goal | Primary Deliverables |
|-------|----------|------|----------------------|
| **Phase 0 – Mobilise** | Weeks 1-2 | Stand up governance, tooling, and success metrics. | RACI map, dashboard of KPIs, backlog triage, delivery playbook. |
| **Phase 1 – Foundations** | Weeks 3-8 | Ship developer experience, typing, and documentation baselines while preparing architecture artefacts. | Dev container, Make targets, onboarding script, architecture diagrams, strict typing policy. |
| **Phase 2 – Reliability & Trust** | Weeks 9-16 | Expand testing, observability, security, and execution coverage. | E2E tests, regression matrix, observability-as-code repo, SBOM pipeline, property-based tests. |
| **Phase 3 – Realism & Performance** | Weeks 17-24 | Close realism gaps in backtest and execution while delivering performance profiling and chaos exercises. | Backtest frictions, pathological scenarios, profiling suite, GameDay runbooks, roadmap refresh. |

_All phases adopt rolling weekly demos and fortnightly retrospectives._

---

## 2. Governance & Tracking

1. **Program RACI**
   - **Sponsor:** CTO (approves scope, removes blockers).
   - **Program Lead:** Head of Platform (maintains roadmap, reports progress).
   - **Workstream Owners:** Assigned per facet (listed below).
   - **Quality Gatekeepers:** QA Lead + Security Lead (sign-off on DoD compliance).

2. **Cadence**
   - Weekly program stand-up (Monday) reviewing burndown, risks, and blockers.
   - Bi-weekly architecture/design review for cross-facet dependencies.
   - Monthly steering committee with leadership to reassess priorities.

3. **Tooling**
   - Track tasks in Jira project `TP-IMPROVE` with swimlanes per facet.
   - Use shared dashboard in Grafana (or Notion) to visualise KPIs (coverage, MTTR, CVE count, etc.).
   - Maintain living roadmap in `docs/roadmap.md`; update after each steering meeting.

4. **Definition of Done (Global)**
   - Code merged to `main` with green CI, documentation updated, owner sign-off.
   - Tests cover new functionality with thresholds met or exceeded.
   - Observability, security, and operations impacts documented.

---

## 3. Integrated Timeline (Week Buckets)

| Workstream → / Week ↓ | 1-2 | 3-4 | 5-6 | 7-8 | 9-10 | 11-12 | 13-14 | 15-16 | 17-18 | 19-20 | 21-22 | 23-24 |
|-----------------------|-----|-----|-----|-----|------|-------|-------|-------|-------|-------|-------|-------|
| Architecture          | Mobilise diagrams tooling | Draft C4, dependency map | Review & publish docs | Plugin framework design | Implement plugin loader | API v1 contracts | Deprecation policy rollout | — | — | — | — | — |
| DevEx                 | Dev container spec | Implement Make targets | Onboarding script | — | — | — | — | — | — | — | — | — |
| Testing               | Coverage gap analysis | Unit test sprints | Enforce coverage gates | E2E pipeline scaffolding | Nightly E2E runs | Regression matrix | — | — | — | — | — | — |
| Observability         | Repo structure for rules | Lint/deploy automation | SLO definition | Healthcheck agents | Alert tuning | Runbook authoring | — | — | — | — | — | — |
| Security              | SBOM tooling PoC | Release integration | SAST/DAST gating | Dependabot rollout | Trivy auto PRs | Policy documentation | — | — | — | — | — | — |
| Execution Engine      | Property-test design | Implement Hypothesis suite | Execution README update | Mocked E2E harness | — | — | — | — | — | — | — | — |
| Backtest              | Friction modelling design | Implement commissions/slippage | Latency/liquidity sims | Pathological scenarios | Validation docs | Real-data comparison | — | — | — | — | — | — |
| Performance           | Profiling scripts | Benchmark harness | Reporting automation | Autotuning design | Autoscaling rollout | — | — | — | — | — | — | — |
| Typing & Style        | Strict mypy config | Lint rule for public APIs | Pre-commit rollout | — | — | — | — | — | — | — | — | — |
| Engineering Culture   | Contributor guide refresh | Label backlog & project board | GameDay plan | GameDay execution | Changelog automation | Roadmap publish | — | — | — | — | — | — |

Legend: cells with `—` indicate workstream idle/monitoring periods.

---

## 4. Workstream Backlogs & Acceptance Criteria

Each backlog table lists discrete deliverables. Create corresponding Jira tickets (`TP-<ID>`).

### 4.1 Architecture & Modularisation (Owner: Principal Architect)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| A1 | Architectural Documentation | Generate C4 context/container diagrams; run `pydeps` for dependency graph; publish under `docs/architecture/`; wire into CI artefacts. | Phase 0 mobilisation | CI job `make docs-architecture` produces diagrams on every merge. | Diagrams referenced in README; last-updated < 30 days. |
| A2 | Pluggable Strategy/Indicator Framework | Design abstract base protocols; implement entrypoint loader; provide sample plugin repo; add contract tests. | A1 | External package installs and registers without modifying core. | Third-party plugin demo passes smoke tests. |
| A3 | API Versioning | Namespace routes, add version headers, generate OpenAPI spec per version, update clients. | A2 | `/api/v1/` deployed; deprecation notice for legacy endpoints documented. | Zero breaking changes reported post-rollout. |

### 4.2 Developer Experience & Environment (Owner: DevEx Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| D1 | VS Code Dev Container | Create `.devcontainer/devcontainer.json`, configure extensions, run post-create scripts. | Phase 0 mobilisation | New hire opens repo in VS Code and runs `make test` successfully. | Onboarding survey: setup < 10 mins median. |
| D2 | Expanded Makefile/CLI | Add `make data-init`, `make codegen`, `make lint-fix`, `make test-all`; document in README. | D1 | CI uses Make targets; docs reference them exclusively. | ≥80% onboarding steps automated via Make. |
| D3 | One-Click Onboarding | Implement `scripts/onboard.sh`; integrate dependency checks, env var scaffolding, smoke tests. | D2 | Script exits 0, provisions sample data, prints summary. | New hires rate onboarding ≥4/5. |

### 4.3 Testing Practices (Owner: QA Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| T1 | Coverage Increase | Identify gaps via `coverage html`; add tests; enforce `--cov-fail-under=80`. | D2 (Make targets) | Coverage gate enabled in CI for `core`, `libs/utils`. | Coverage dashboard shows ≥80% sustained. |
| T2 | End-to-End Scenarios | Build docker-compose stack; create ingestion→order tests; schedule nightly run. | T1 | Nightly E2E pipeline green; artifacts stored. | Mean E2E pass rate ≥95%. |
| T3 | Regression Matrix | Create `tests/TEST_PLAN.md`; link automated tests; integrate into release checklist. | T2 | Release PR template references matrix. | 100% release PRs include matrix link. |

### 4.4 Observability (Owner: Observability Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| O1 | Observability-as-Code | Organise repo directories; lint dashboards; add CI validation. | Phase 0 mobilisation | `make obs-validate` passes; dashboards versioned. | Deployment rollback possible via git tags. |
| O2 | Health Checks & SLO Alerts | Define SLOs; implement synthetic probes; configure alert rules. | O1 | Alerts deployed with runbook references. | MTTA < 5 min, MTTR trending downward. |
| O3 | Monitoring Documentation | Update `docs/observability.md` with runbooks, diagrams. | O2 | On-call exercise completes using docs only. | Post-incident survey satisfaction ≥4/5. |

### 4.5 Security Processes (Owner: Security Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| S1 | SBOM Pipeline | Integrate `syft`/`cyclonedx` into release; publish artifacts. | Phase 0 mobilisation | Release pipeline attaches SBOM to GitHub Releases. | 100% releases include SBOM. |
| S2 | Vulnerability Gates | Configure CodeQL, ZAP/DAST; enforce branch protections; document triage. | S1 | Merge blocked on critical vulns; exception process documented. | Zero critical vulns merged without waiver. |
| S3 | Dependency Hygiene | Enable Dependabot, Trivy scanning; auto-close fixed PRs. | S2 | CVE dashboard auto-updates; notifications routed to #security channel. | Critical CVEs remediated <7 days median. |

### 4.6 Execution Engine (Owner: Execution Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| E1 | Property-Based Tests | Define Hypothesis strategies; integrate invariants; add CI job. | T1 | Tests run in CI and fail on invariant violation. | Bug discovery rate increases ≥1/month. |
| E2 | Advanced Order Documentation | Update `execution/README.md`; document partial fills, cancel/replace; raise issues for gaps. | E1 | README versioned; gaps tracked in Jira with owners. | Stakeholder sign-off on coverage. |
| E3 | Mocked Functional E2E | Build mocked exchange fixtures; validate routing/risk. | E1, T2 | CI pipeline step `make execution-e2e` passes without external deps. | 100% PRs touching execution run suite. |

### 4.7 Backtest Engine Realism (Owner: Quant Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| B1 | Market Frictions | Implement configurable commissions, slippage, latency, liquidity; store configs. | E1 | Backtester exposes parameters with docs. | Simulated PnL within ±5% of prod. |
| B2 | Pathological Scenarios | Create flash crash, gap, halt scenarios; document expected behaviour. | B1 | `docs/backtest_scenarios.md` published. | All strategies must pass gating scenarios. |
| B3 | Real Data Benchmark | Compare live logs vs. simulation; calibrate parameters; publish report. | B2 | Report stored under `reports/backtest_validation/`. | Quarterly validation cadence maintained. |

### 4.8 Performance & Parallelism (Owner: Performance Engineer)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| P1 | Profiling Suite | Build CPU/memory profiling scripts; document usage; integrate flamegraph tooling. | D2 | `make profile-cpu` & `make profile-mem` produce artefacts. | Profiling run monthly with report. |
| P2 | Latency & Throughput Benchmarks | Implement benchmarking harness; capture baseline metrics; store under `reports/performance/`. | P1 | Benchmark CI job compares regressions; alerts on >5% degradation. | Latency/throughput budgets defined and tracked. |
| P3 | Auto-Tuning/Scaling | Implement adaptive worker logic; configure HPA/Compose scale. | P2 | Production load tests confirm scaling to SLO thresholds. | 0 incidents due to resource saturation post-rollout. |

### 4.9 Typing & Style (Owner: Tooling Lead)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| Y1 | Strict Typing | Enable `mypy --strict` for key packages; add `py.typed`. | D2 | CI fails on typing regressions. | Strict typing coverage ≥90% modules. |
| Y2 | Public API Type Hints | Add lint rule; update guidelines. | Y1 | Lint gate enforces annotations on exports. | Zero new public APIs without hints. |
| Y3 | Automated Formatting & Linting | Configure pre-commit with `black`, `ruff`, `isort`, `mypy`, `gofmt`; update docs. | Y2 | `pre-commit run --all-files` clean; CI uses same hooks. | Formatting-related review comments drop to near-zero. |

### 4.10 Engineering Culture & Activity (Owner: Engineering Manager)

| ID | Initiative | Key Tasks | Dependencies | Definition of Done | Success Metric |
|----|------------|-----------|--------------|--------------------|----------------|
| C1 | External Contributor Enablement | Refresh `CONTRIBUTING.md`; add templates; tag issues; launch project board. | Phase 0 mobilisation | External contributors successfully submit PR using guide. | Unique external contributors +50% QoQ. |
| C2 | GameDay & Chaos Testing | Develop scenarios; run staging GameDay; capture learnings. | O2, E3 | Postmortem published with follow-up actions. | Mean incident response KPIs improve ≥20%. |
| C3 | Changelog & Roadmap Hygiene | Automate changelog via `towncrier` or `cz`; publish quarterly roadmap. | C1 | Release cycle includes changelog PR; roadmap shared with stakeholders. | Stakeholder satisfaction survey ≥4/5. |

---

## 5. Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Competing product roadmap work reduces available capacity. | Slips in phased timeline. | Medium | Secure executive sponsorship; allocate dedicated capacity per squad. |
| Tooling sprawl from new scripts and CI jobs. | Maintenance overhead; flaky pipelines. | Medium | Establish owner per tool; run quarterly tool audits. |
| Alert fatigue due to aggressive SLO thresholds. | On-call burnout; ignored alerts. | Low | Pilot thresholds in staging; iterate before production rollout. |
| Plugin framework introduces security vectors. | Supply-chain risk. | Medium | Mandate signing/allowlist for plugins; security review before enabling. |

---

## 6. Reporting & Success Measurement

- **Weekly Metrics:** test coverage, build success rate, open CVEs, MTTR, onboarding time.
- **Quarterly Review:** compare against target maturity scores; adjust backlog with leadership.
- **Exit Criteria:** all success metrics trending at or above targets, stakeholder survey ≥9/10, zero critical audit findings outstanding.

---

_Review and update this program plan every quarter. Archive superseded versions under `docs/history/improvement_plan/` with change logs for traceability._
