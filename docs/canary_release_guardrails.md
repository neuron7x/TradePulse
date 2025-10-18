# Canary Release Guardrails for TradePulse Deployments

## Purpose and Scope
This playbook defines the safety rails for promoting new versions of TradePulse services through staged, canary-based deployments. It aligns production risk management with business outcomes by specifying:

- Quantitative thresholds for market-facing impact (PnL volatility, order execution latency, error budgets).
- Technical guardrails that drive automated rollback and halt criteria.
- Observability, alerting, and governance hooks that ensure every release decision is auditable.

The guidance applies to every production-bound deployment in equities, derivatives, and crypto execution clusters. Teams must adapt cluster-specific parameters but cannot relax the mandatory guardrails without governance approval.

## Release Phases and Traffic Allocation
| Phase | Traffic Share | Duration | Objectives |
|-------|---------------|----------|------------|
| Pre-flight validation | 0% (shadow only) | 15 minutes minimum | Replay latest real market feed to the new build. Validate basic health checks, schema compatibility, and warm caches. |
| Canary Phase 1 | 1% of market-making orders, 0.5% of agency flow | 20 minutes minimum | Confirm no regression in core health KPIs vs. control. Enable detailed tracing and per-order logging. |
| Canary Phase 2 | 5% of market-making, 2% of agency flow | 30 minutes minimum | Exercise broader order types and venues. Evaluate PnL, latency, and reject ratios against guardrails. |
| Canary Phase 3 | 15% of market-making, 5% of agency flow | 45 minutes minimum | Validate stability under burst traffic scenarios. Confirm guardrails remain within green band. |
| General Availability | 100% | Until next release | Promote once all canary checkpoints pass, change tickets closed, and sign-off recorded. |

**Rules:**

- Canary routing is enforced via the deployment controller (see `deploy/traffic_splitter.yaml`) with immutable manifests checked into Git.
- Traffic shares apply at the per-venue level; never co-mingle canary and baseline orders within the same market data session.
- Deviations from the schedule require incident-commander authorization logged in the release record.

## Technical Guardrails and Thresholds
All metrics are computed as rolling windows over the phase duration with 2-minute sampling. Thresholds marked "hard" trigger an immediate rollback; "soft" requires manual review and extension of the canary window.

### Core Stability Metrics
| Metric | Baseline Source | Soft Threshold | Hard Threshold | Notes |
|--------|-----------------|----------------|----------------|-------|
| Order acceptance error rate | `execution.order_accept.failures` | +25% over control | +40% over control | Compare to matched baseline traffic. |
| Risk engine latency (p95) | `risk.latency.p95_ms` | +15% | +25% | Latency measured pre-trade. |
| Matching latency (p99) | `execution.match.latency.p99_ms` | +10% | +20% | Venue-specific; apply tightest venue limit. |
| Quote staleness (p50) | `markets.quotes.age_ms` | +15% | +30% | Monitored per symbol bucket. |
| CPU saturation | `infra.host.cpu_utilization` | 80% | 90% | Hard breach fires auto rollback. |
| Memory headroom | `infra.host.mem_available_pct` | 20% | 10% | Hard breach fires auto rollback. |

### Business Impact Metrics
| Metric | Baseline Source | Soft Threshold | Hard Threshold | Notes |
|--------|-----------------|----------------|----------------|-------|
| Realized PnL delta | `pnl.realized.delta_bps` | ±5 bps | ±8 bps | Compute vs. control bucket normalized for exposure. |
| Mark-to-market drift | `pnl.unrealized.drift_bps` | ±4 bps | ±6 bps | Adjust for market beta using hedging desk factor model. |
| Missed spread capture | `execution.spread_capture.effective_bps` | -5% | -8% | Applies to market making strategies only. |
| Agency order fill ratio | `execution.agency.fill_ratio` | -3% | -5% | Weighted by client priority tier. |

### Data Quality and Risk Controls
- **Schema Guardrails:** Any breaking change detected by schema diff (`schemas/*`) aborts deployment pre-flight.
- **Reference Data Freshness:** `data.refdata.age_minutes` must remain below 5 minutes; breach triggers rollback.
- **Model Drift:** Canary uses `analytics/model_guardian.py` checks; drift score > 0.35 halts promotion.

## Automated Rollback Logic

1. Deployment controller subscribes to guardrail metrics via Prometheus Alertmanager webhooks.
2. When a hard threshold fires, `deploy/canary_guardian.py` executes:
   - Freeze new order intake for canary pods (via Kubernetes PodDisruptionBudget adjustment).
   - Drain existing sessions gracefully (timeout 10 seconds) to avoid orphaned orders.
   - Redeploy baseline revision (`kustomize build deploy/overlays/prod | kubectl apply -f -`).
   - Record rollback reason, metric snapshot, and operator on-call acknowledgement in `reports/release_decisions/`.
3. Soft threshold alerts route to the release commander Slack channel (`#deploy-ops`). Manual override requires documented rationale in the release decision log.

Rollback automation is tested quarterly in the chaos game day (see `docs/resilience.md`).

## Observability and Alerting Integration

- **Dashboards:** Grafana folder `TradePulse/Canary` hosts phase-specific dashboards with synchronized time ranges for baseline vs. canary comparisons.
- **Alerting:**
  - Critical guardrails use PagerDuty service `tradepulse-prod` with a 5-minute auto-escalation.
  - Warning-level guardrails trigger Slack and create Jira ticket via OpsGenie integration.
- **Tracing:** Enable 100% sampling for canary pods via `observability/tracing_config.yaml`. Retain spans for 72 hours.
- **Log Enrichment:** Canary pods append `deployment_id`, `phase`, and `traffic_share` fields to structured logs to support post-mortems.

## Decision Logging and Audit Trail

Every phase transition requires updating `reports/release_decisions/<deployment_id>.yaml` with:

```yaml
phase: canary_phase_2
approver: jane.doe@tradepulse.ai
start_timestamp: 2024-05-16T21:15:00Z
traffic_share:
  market_making: 0.05
  agency: 0.02
metrics_snapshot:
  execution.match.latency.p99_ms: 6.8
  pnl.realized.delta_bps: 1.1
notes: "Latency normalized after cache warmup. Proceeding to phase 3."
```

Decision logs are immutable once GA is reached and retained for at least 12 months.

## Safe Release Reporting

- **Daily Canary Report:** Generated automatically by `reports/generate_canary_report.py`. Summarizes guardrail adherence, breaches, and rollback outcomes. Distributed to trading ops, risk, and product leads.
- **Monthly Safety Review:** Aggregates canary performance across releases, highlights recurring guardrail breaches, and updates threshold proposals.
- **Regulatory Archive:** Export sanitized reports and decision logs to the compliance vault (`infra/compliance/retention_bucket`) within 24 hours of GA.

## Governance and Continuous Improvement

- Guardrail thresholds are reviewed quarterly with stakeholders from trading, risk, and SRE. Updates require sign-off recorded in `governance/change_log.md`.
- Any production incident attributable to a guardrail gap triggers an ADR to refine the framework.
- Teams must document experiments that propose loosening thresholds, including Monte Carlo backtests demonstrating risk neutrality.

Adhering to these guardrails ensures canary releases protect both client outcomes and TradePulse's PnL while enabling rapid, safe iteration.
