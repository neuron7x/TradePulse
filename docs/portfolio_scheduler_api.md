# Portfolio Optimisation, Scheduling, and API Contracts

This guide consolidates the engineering standards for three critical layers in
TradePulse: portfolio optimisation research, scheduler orchestration for data
pipelines, and external service APIs. The goal is to ensure that optimisation
experiments, recurring refreshes, and runtime integrations follow a uniform,
auditable process.

---

## 1. Portfolio Optimisation Baselines

| Topic | Standard | Implementation Notes |
| --- | --- | --- |
| **Cost Function** | Document the transaction cost and slippage assumptions alongside each strategy. | Capture the model description in the strategy documentation (`docs/scenarios.md`) and link to the experiment artefacts stored in `reports/`.
| **Optimisation Problem** | Make the optimisation objective and constraints reproducible. | Commit solver notebooks or scripts under `docs/notebooks/` or `examples/` so the process can be replayed by reviewers.
| **Decision Variables** | Clearly describe the inputs/outputs that flow into strategy code. | Align variable naming with the helpers in `core/strategies/` and surface expected shapes or domains in docstrings.
| **Constraints** | Keep leverage, turnover, and exposure guardrails explicit. | Use configuration parameters in `configs/` (see `configs/README.md`) and reference them from the associated strategy module.
| **Data Contracts** | Reference validated datasets when running simulations. | Point to the canonical sources described in `docs/dataset_catalog.md` and note any preprocessing carried out by `scripts/data_sanity.py`.
| **Validation** | Pair each optimisation change with a measurable backtest. | Follow the workflow in `docs/backtest.md` and publish results in `reports/` for auditability.

### 1.1 Configuration

- Store optimisation-specific parameters alongside existing configuration files
  in `configs/` (for example `configs/amm_strategy.yaml`). Use schema comments in
  the file to describe each field.
- Keep solver defaults near the corresponding strategy entry point in
  `core/strategies/` so code and configuration evolve together.

### 1.2 Testing Matrix

| Test | Purpose | Frequency |
| --- | --- | --- |
| **Unit Coverage** | Exercise numerical helpers and constraint builders. | `pytest tests/unit -k optimisation` before publishing a change. |
| **Scenario Regression** | Compare portfolio metrics for representative data slices. | Nightly or on-demand via the standard `pytest` workflows documented in `TESTING.md`. |
| **Governance Review** | Summarise TC assumptions, performance, and risk. | Monthly governance checkpoint with artefacts attached in `reports/`. |

---

## 2. Scheduler: Ingestion, Backfills, and Reports Orchestration

| Capability | Requirement | Tooling |
| --- | --- | --- |
| **Workflow Overview** | Describe the orchestration approach that triggers data, backfills, and reporting. | Maintain the narrative in `docs/monitoring.md` and extend it when adding new recurring tasks. |
| **SLA Tracking** | Define expected completion windows for critical pipelines. | Capture SLA thresholds in configuration or documentation and monitor via the metrics outlined in `observability/metrics.json`. |
| **Retries & Alerts** | Capture how failures are surfaced to operators. | Reuse alerting conventions from `observability/alerts.json` and ensure escalation policies are documented. |
| **Dependency Graph** | Make ordering between pipelines explicit. | Document dependencies in `docs/scenarios.md` (ingestion → backfill → reporting) and link to implementation scripts or runbooks. |
| **Backfill Windows** | Record the criteria used to replay historical data. | Use the checks in `scripts/data_sanity.py` and document parameter ranges alongside the backfill procedure. |

### 2.1 Operational Runbooks

1. **Ingestion Tasks** – Capture the cadence and data sinks in the runbook. SLA:
   2 minutes end-to-end unless otherwise noted.
2. **Backfill Tasks** – Triggered manually or by anomaly detection. Describe
   validation steps prior to replaying missing windows.
3. **Reporting Tasks** – Aggregate metrics for stakeholders after successful
   ingestion and backfill cycles.

### 2.2 Scheduler Testing & Observability

- Add integration coverage under `tests/integration/` when scheduler logic is
  extended.
- Use synthetic failure drills to ensure alerts contain task identifiers,
  timestamps, and exception summaries.
- Dashboards should highlight SLA compliance, retry counts, and queue depth by
  reusing the Grafana layouts in `observability/grafana/`.

---

## 3. Rolling Feature Windows and Backtest Completeness Guards

| Control | Description | Implementation |
| --- | --- | --- |
| **Rotational Windows** | Shift feature refresh windows based on market sessions to minimise overlap. | Describe the rotation schedule in `docs/feature_store_sync_and_registry.md` and reference the controlling configuration values in `configs/`. |
| **Completeness Check** | Validate feature availability before launching backtests. | Use the feature registry guidance in `docs/feature_store_sync_and_registry.md` and enforce checks in pre-launch scripts. |
| **Staging Snapshot** | Produce immutable snapshots for reproducibility. | Store snapshot metadata or summaries within the `reports/` directory using timestamped filenames. |
| **Audit Trail** | Log decisions made during window rotation and completeness checks. | Persist structured logs via the observability helpers in `observability/health.py` and archive outputs with other operational reports. |

### 3.1 Backtest Launch Checklist

1. Confirm latest rotation window applied (verify against documentation or the
   relevant dashboard).
2. Run completeness verification; archive the success receipt alongside the
   backtest request notes in `reports/`.
3. Lock manifest references and feature snapshot identifiers into the job
   payload.
4. Notify the strategy owner with a summary before execution.

---

## 4. API Contracts and Client Generation

| Aspect | Requirement | Notes |
| --- | --- | --- |
| **Specification** | Publish OpenAPI or JSON schema definitions with the service change. | Place artefacts in `docs/schemas/` next to the existing schema index and link them from `docs/integration-api.md`. |
| **Contract Testing** | Ensure the runtime implementation adheres to the published schema. | Add targeted tests under `tests/integration/` and run `pytest` locally before pushing. |
| **Client Generation** | Keep generated clients in sync with contract updates. | Document the regeneration steps (e.g., scripts or notebooks) and commit artefacts alongside the schema update. |
| **Strict Validation** | Enable request/response validation in services. | Reference the validation helpers available in `core/` modules and document behaviour in the service README. |
| **Versioning** | Follow semantic versioning and note deprecations. | Capture changes in `reports/release_readiness.md` and communicate timelines to downstream users. |

### 4.1 Developer Workflow

1. Update the API definition in `docs/schemas/` and cross-link from
   `docs/integration-api.md`.
2. Regenerate clients using the documented scripts or notebooks and commit the
   outputs.
3. Execute the relevant `pytest` suites locally before pushing changes.
4. Ensure the changelog entry references the affected endpoints.

---

## 5. API Rate Limiting and Backpressure Controls

| Control | Policy | Implementation Details |
| --- | --- | --- |
| **Token-Bucket Rate Limiter** | Establish service-specific throughput limits and document them. | Capture quotas in service configuration or README files and expose utilisation metrics using `observability/metrics.json`. |
| **N+1 Protection** | Detect fan-out queries that risk overwhelming dependencies. | Instrument client wrappers with counters or gauges and describe the rollout in `docs/monitoring.md`. |
| **Backpressure** | Propagate headers or status codes that tell clients to slow down. | Document supported headers/statuses in `docs/integration-api.md` and ensure clients honour the policy. |
| **Adaptive Scaling** | React to sustained utilisation by scaling horizontally. | Integrate scaling policies with deployment manifests documented in `deploy/` and surface telemetry through the observability stack. |
| **Audit Logging** | Log rate limiting decisions with structured context. | Use the logging conventions described in `observability/README.md` and archive daily summaries alongside other operational reports. |

### 5.1 Testing & Monitoring

- Add load or soak tests under `tests/performance/` when evolving rate limiting
  policies.
- Monitor the health dashboards in `observability/grafana/` for rate limiting
  metrics and alert fatigue.
- Ensure incidents referencing rate limiting controls are tracked in
  `reports/` with follow-up actions.

---

By codifying these standards, TradePulse maintains consistent optimisation
results, predictable data refresh pipelines, and reliable external interfaces.
