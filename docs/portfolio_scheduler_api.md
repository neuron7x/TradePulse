# Portfolio Optimisation, Scheduling, and API Contracts

This guide consolidates the engineering standards for three critical layers in
TradePulse: portfolio optimisation with non-linear transaction cost (TC)
modelling, scheduler orchestration for data pipelines, and external service
APIs. The goal is to ensure that optimisation research, data refreshes, and
runtime integrations follow a uniform, auditable process.

---

## 1. Portfolio: Non-linear Transaction Cost Models with CVXPy Backend

| Topic | Standard | Implementation Notes |
| --- | --- | --- |
| **Cost Function** | Model slippage and market impact with piecewise-quadratic or exponential cost curves expressed as convex functions. | Encode cost curve as `cvxpy.Function` composed of quadratic segments or exponential envelopes; maintain differentiability to keep solver stable. |
| **Optimisation Problem** | Formulate as convex program minimising expected loss = risk term + TC penalty - alpha. | Use CVXPy canonical form; solver defaults to ECOS for quadratic, SCS for exponential. Configure warm-start with last feasible portfolio to accelerate convergence. |
| **Decision Variables** | Optimise both dollar weights and trade notional. | Introduce variable `delta_w` for trade increments; enforce `positions_{t+1} = positions_t + delta_w`. |
| **Constraints** | Enforce leverage, turnover, and sector exposure limits. | Use `cvxpy.norm1(delta_w) <= turnover_limit` and group exposure constraints keyed by `sector_id`. |
| **Data Contracts** | Require inputs from `domain/portfolio/models.py` manifest to ensure consistent factor loadings and borrow costs. | Publish manifest hash alongside optimisation job metadata; fail job if hash mismatch. |
| **Validation** | Run calibration backtest verifying realised TC vs. model forecast within ±5% over trailing 30 days. | Persist validation artifacts to `reports/portfolio/tc_validation/<date>.md`. |

### 1.1 Configuration

- Store TC curve parameters in `configs/portfolio/tc_curves.yaml` with schema:
  ```yaml
  symbol: <ticker>
  liquidity_bucket: <enum>
  impact_model: piecewise_quadratic|exponential
  parameters:
    - { breakpoint: 25000, slope: 0.8, quad: 0.02 }
  ```
- Use `core.optimisation.tc_loader.load_curves(...)` to deserialize and feed into
  CVXPy model construction.
- Register solver settings in `configs/portfolio/optimisation.yaml` (tolerances,
  max iterations, warm-start flags).

### 1.2 Testing Matrix

| Test | Purpose | Frequency |
| --- | --- | --- |
| **Solver Regression** | Execute deterministic portfolio optimisation scenario comparing output weights against golden snapshot. | Nightly CI (`make optimisation-test`). |
| **Stress Scenario** | Run scenario with widened bid/ask spreads to ensure solver remains feasible. | Weekly research review. |
| **PnL Attribution Audit** | Compare TC-adjusted PnL vs. raw PnL to validate impact modelling. | Monthly governance checkpoint. |

---

## 2. Scheduler: Ingestion, Backfills, and Reports Orchestration

| Capability | Requirement | Tooling |
| --- | --- | --- |
| **Workflow Engine** | Support both Airflow and Prefect runners via adapter pattern. | Implement `schedulers/adapter.py` exposing `submit_dag(...)` that delegates to Airflow REST API or Prefect client based on config. |
| **SLA Tracking** | Declare per-DAG SLA in metadata (`sla_minutes`) and monitor via Prometheus metric `tradepulse_scheduler_sla_breach_total`. | Emit metric from adapter when completion exceeds SLA. |
| **Retries & Alerts** | Configure exponential backoff with max 5 attempts; final failure routes to `#quant-ops` Slack webhook. | Use Airflow retry policy or Prefect `RetryPolicy`; integrate with `observability/alerting.py`. |
| **Dependency Graph** | Express DAG dependencies explicitly; ingestion must complete before backfills, backfills before reports. | Use Airflow `ExternalTaskSensor` or Prefect `wait_for`. Maintain dependency manifest in `configs/scheduler/dependencies.yaml`. |
| **Backfill Windows** | Allow parameterised replay windows (`start_ts`, `end_ts`) sourced from data completeness checks. | Validate windows via `data_quality.check_window_complete(...)` prior to submission. |

### 2.1 Operational Runbooks

1. **Ingestion DAG** – Runs every minute, lands raw market data into bronze
   storage. SLA: 2 minutes end-to-end.
2. **Backfill DAG** – Triggered manually or by anomaly detection. Validates data
   completeness before replaying missing windows.
3. **Report DAG** – Aggregates metrics for stakeholders post-backfill. Requires
   both ingestion and backfill success statuses.

### 2.2 Scheduler Testing & Observability

- Integration test `tests/scheduler/test_dependency_contracts.py` asserts that
  generated DAGs respect declared dependencies.
- Synthetic failure harness triggers retry paths weekly; alerts must include DAG
  id, run id, and exception summary.
- Dashboard `Scheduler Health` charts SLA compliance, retry counts, and queue
  depth (`tradepulse_scheduler_queue_depth`).

---

## 3. Rolling Feature Windows and Backtest Completeness Guards

| Control | Description | Implementation |
| --- | --- | --- |
| **Rotational Windows** | Shift feature refresh windows based on market sessions (e.g., APAC, EMEA, AMER) to minimise overlap. | Define rotation schedule in `configs/features/window_rotation.yaml`; scheduler selects appropriate window per session. |
| **Completeness Check** | Validate all feature tables for target horizon before launching backtest. | Run `feature_registry.verify_completeness(manifest_hash, window)`; block run if any feature is stale or missing partitions. |
| **Staging Snapshot** | Produce immutable snapshot of feature data for reproducibility. | Store snapshot metadata in `reports/backtest/snapshots/<date>.json`. |
| **Audit Trail** | Log rotation and completeness decisions. | Append events to `observability/audit/feature_windows.jsonl` with fields `{timestamp, session, window, completeness_status}`. |

### 3.1 Backtest Launch Checklist

1. Confirm latest rotation window applied (check scheduler dashboard or audit
   log).
2. Run completeness verification; ensure success receipt stored alongside
   backtest request.
3. Lock manifest hash and feature snapshot ID into backtest job payload.
4. Notify strategy owner with summary before execution.

---

## 4. API Contracts and Client Generation

| Aspect | Requirement | Notes |
| --- | --- | --- |
| **Specification** | Every external service publishes OpenAPI 3.1 definition under `schemas/openapi/<service>.yaml`. | Use shared templates for pagination, error envelopes, and authentication headers. |
| **Contract Testing** | CI job `make api-contract-verify` ensures server implementation matches specification via Schemathesis. | Include negative tests for validation errors (missing fields, type mismatches). |
| **Client Generation** | Auto-generate Python and TypeScript clients during build pipeline. | Use `openapi-python-client` and `openapi-typescript-codegen`; publish to internal artifact registry. |
| **Strict Validation** | Enable request/response validation middleware in services (`strict_mode=true`). | Reject unknown fields; return structured error with trace id. |
| **Versioning** | Adopt semantic versioning; breaking changes require new major version path (`/v2`). | Document deprecations in `docs/release-notes/api.md`. |

### 4.1 Developer Workflow

1. Update OpenAPI definition.
2. Run `make api-clients` to regenerate Python/TS clients.
3. Execute contract tests locally before pushing.
4. Commit spec + generated clients; ensure changelog entry references endpoint.

---

## 5. API Rate Limiting and Backpressure Controls

| Control | Policy | Implementation Details |
| --- | --- | --- |
| **Token-Bucket Rate Limiter** | Default: 100 requests/second burst with 500 token bucket, refill 100 tokens/sec. | Implement via shared `middlewares/rate_limit.py`; configure thresholds per service in `configs/api/rate_limits.yaml`. |
| **N+1 Protection** | Detect fan-out queries exceeding 20 downstream requests. | Instrument client wrappers to emit `tradepulse_api_fanout_total`; trigger circuit breaker when threshold exceeded. |
| **Backpressure** | Propagate `Retry-After` headers and gRPC status `RESOURCE_EXHAUSTED`. | Upstream services must respect headers and apply exponential backoff with jitter. |
| **Adaptive Scaling** | Auto-scale API pods when sustained utilisation >70% for 5 minutes. | Use HPA metrics fed by rate limiter queue depth. |
| **Audit Logging** | Log rate limiting decisions with structured context (client id, quota, action). | Write to `observability/audit/api_rate_limit.jsonl`; rotate daily. |

### 5.1 Testing & Monitoring

- Load test scenario `bench/api_rate_limit_test.py` validates limiter behaviour
  under burst traffic.
- Prometheus metrics: `tradepulse_api_rate_limit_hits_total`,
  `tradepulse_api_backpressure_active`, `tradepulse_api_nplus1_block_total`.
- Alerts: breach of N+1 threshold opens incident ticket via PagerDuty policy
  `API-Backpressure`.

---

By codifying these standards, TradePulse maintains consistent optimisation
results, predictable data refresh pipelines, and reliable external interfaces.
