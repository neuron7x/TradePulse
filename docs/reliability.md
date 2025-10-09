# Reliability Targets, SLOs, and Escalation Policy

This playbook aligns TradePulse engineering and operations teams on the
service-level commitments we communicate to stakeholders. It converts product
expectations into actionable SLOs with explicit error budgets, alerting rules,
and escalation paths so reliability work can be prioritised alongside feature
work.

## Scope and Ownership

| Domain | Services | Primary Owners | Supporting Teams |
| ------ | -------- | -------------- | ---------------- |
| Client API | REST/GraphQL edge, authentication, request fan-out | Execution Platform | Infrastructure, SRE |
| Market Data | Real-time ingestion, historical snapshots, feature stores | Data Platform | Infrastructure |
| Strategy Runtime | Signal evaluation, backtest coordinator, portfolio engine | Quant Engineering | SRE |
| Order Execution | Broker adapters, risk guards, position reconciliation | Execution Platform | Compliance, Infrastructure |

Each domain lead is accountable for SLO definitions, observing error budget
consumption, and initiating corrective actions. SRE facilitates the review
cadence, provides tooling support, and is the final approver for changes that
impact shared infrastructure or customer-visible reliability.

## SLA Commitments

TradePulse publishes the following externally visible SLAs. SLOs must be set
with sufficient safety margin to guarantee the SLA when measured over a rolling
90-day window.

| Capability | SLA Metric | Customer Commitment |
| ---------- | ---------- | ------------------- |
| Client API availability | Successful request ratio | ≥ 99.5% | 
| Strategy order latency | Time from order submit to broker acknowledgement | ≤ 400 ms for 95% of orders |
| Market data freshness | Delay between exchange event and availability via API | ≤ 3 seconds 99% of the time |

Breaching an SLA triggers incident review with product and customer success and
may require service credits per contractual terms.

## Service-Level Objectives

Internal SLOs are tuned tighter than SLAs to preserve buffer. Objectives are
tracked weekly and trended quarterly.

| Service | SLI Definition | Target | Measurement Window |
| ------- | -------------- | ------ | ------------------ |
| Client API | Ratio of 2xx/3xx responses to total requests, excluding 4xx | 99.9% availability | Rolling 30 days |
| Client API latency | p95 end-to-end latency for `/orders` and `/positions` | ≤ 250 ms | 5-minute sliding windows |
| Strategy runtime | Successful job completions / attempted jobs | 99.7% | Rolling 7 days |
| Strategy runtime latency | p99 time to produce decision from signal batch | ≤ 120 ms | 1-hour buckets |
| Order execution | Orders confirmed within broker SLA / total orders | 99.9% | Rolling 30 days |
| Market data freshness | Percentage of ticks arriving < 1.5 s from event time | 99.8% | Rolling 24 hours |
| Data pipeline accuracy | Jobs with parity checks passing / total jobs | 99.95% | Rolling 30 days |

Targets assume at least 1,000 valid events per window; otherwise the period is
flagged for manual review.

### Instrumentation & Measurement

- **Data sources** – All SLIs must be captured from authoritative telemetry.
  - Availability and latency SLIs are sourced from the `edge_requests_total`
    and `edge_request_duration_seconds` Prometheus metrics scraped from the API
    gateways.
  - Strategy runtime SLIs rely on the `workflow_job_duration_seconds` histogram
    emitted by the orchestrator alongside job outcome tags.
  - Market data freshness is computed by comparing exchange timestamps against
    the `market_tick_ingest_timestamp` gauge produced by the ingestion workers.
- **PromQL references** – Dashboards and alert rules should rely on explicit
  queries to ensure consistency across environments. Example queries:
  - `sum(rate(edge_requests_total{status=~"2..|3.."}[5m])) / sum(rate(edge_requests_total[5m]))`
  - `histogram_quantile(0.95, sum(rate(edge_request_duration_seconds_bucket{route=~"/orders|/positions"}[5m])) by (le))`
  - `1 - (sum(increase(workflow_job_failures_total[1h])) / sum(increase(workflow_job_attempts_total[1h])))`
- **Data retention** – Prometheus retains 30 days of raw samples. Aggregated
  SLI summaries are exported nightly to BigQuery for historical trend analysis
  and quarterly review decks.
- **Gaps & overrides** – If telemetry is unavailable for more than two
  consecutive measurement windows, the owning domain must populate an incident
  ticket explaining the gap and manually reconstruct the SLI from logs or
  customer impact reports.

### Error Budget Policy

Error budgets are derived as `(1 - SLO target)` per window. The following guard
rails apply:

- **Green (< 25% consumed)** – Continue regular releases. Document regression
  tests relevant to observed risks.
- **Yellow (25–75% consumed)** – Require SRE sign-off for deploys touching the
  affected service. Increase sampling on synthetic probes and ensure alert run
  books are updated.
- **Red (> 75% consumed)** – Freeze non-critical deploys. Run a game-day or
  chaos exercise targeting the affected component within two weeks. Schedule a
  postmortem review with engineering and product leadership.

Error budget burn rates are evaluated hourly using Grafana burn-rate panels and
Prometheus recording rules. Alerts fire when projected exhaustion is under 72
hours for Red services or 7 days for Yellow services.

### Release Gates & Change Management

- **Deploy approvals** – Any deploy that touches a service in Yellow or Red
  status requires sign-off from both the domain lead and the on-call SRE. The
  approving engineer documents mitigation steps in the deployment ticket.
- **Production experiments** – Feature flags and shadow traffic must include a
  rollback condition tied to the relevant SLI. Experiments that consume more
  than 10% of the error budget in a week are automatically disabled by the flag
  service.
- **Maintenance windows** – Planned downtime must be announced 7 days in
  advance and includes a mitigation plan referencing the impacted SLOs. Any
  window that risks breaching SLA targets requires VP Engineering approval.
- **Change freeze** – A platform-wide freeze is enacted the week before major
  regulatory filings or earnings announcements. Exceptions require a SEV-1
  mitigation and must be logged in the reliability changelog.

## Alerting and Escalation

### Severity Ladder

| Severity | Trigger Examples | Response Expectation |
| -------- | ---------------- | -------------------- |
| SEV-1 | SLA breach in progress, sustained 30-min outage, data corruption | Page on-call immediately, incident commander within 5 minutes, notify leadership within 15 minutes |
| SEV-2 | SLO violation projected within 24 hours, partial feature outage | Page on-call within 15 minutes, engage domain owner, customer updates every hour |
| SEV-3 | Degraded performance without customer impact, tooling failure | Create Jira ticket, respond in business hours, update status weekly |

### Escalation Flow

1. **Detection** – Prometheus alerts, synthetic probes, or support cases detect
   an issue and route to PagerDuty (`TradePulse/SRE` schedule).
2. **On-call response** – On-call acknowledges within 5 minutes (SEV-1/2) or the
   next business hour (SEV-3). They open an incident channel (`#inc-YYYYMMDD`) and
   start an incident log.
3. **Escalation** – If unresolved within 15 minutes (SEV-1) or 60 minutes
   (SEV-2), page the domain engineering manager. Persistent degradation triggers
   escalation to VP Engineering and Product lead.
4. **Communications** – Customer success posts updates to the status page every
   30 minutes for SEV-1 and hourly for SEV-2. Post-incident report is due within
   48 hours of resolution.

### Incident Lifecycle & Postmortems

- **Stabilise** – The incident commander coordinates triage, ensures mitigations
  are tracked in the incident log, and designates an on-call scribe.
- **Document** – Within 12 hours of resolution, the commander files a
  lightweight summary including timeline, blast radius, and remaining risks.
- **Analyse** – A blameless postmortem is required for SEV-1 or any SEV-2 that
  breaches an SLO. Use the [`docs/postmortem_template.md`](postmortem_template.md)
  contributing factors, and identify action items with clear owners and due
  dates.
- **Follow-through** – Action items enter the Reliability Kanban board. The SRE
  lead reviews status weekly until closure; overdue tasks are escalated in the
  reliability governance meeting.

## Governance and Review Cadence

- **Weekly** – Review error budget dashboard in SRE sync, assign owners to
  investigate burn trends.
- **Monthly** – Domain leads refresh SLO definitions and validate alert
  thresholds. Update this document with changes and circulate meeting notes.
- **Quarterly** – Leadership reviews SLA adherence, approves budget for
  reliability initiatives, and signs off on any SLA changes.

### Reporting & Scorecards

- **Reliability scorecard** – Updated the first business day of each month in
  `reports/reliability_scorecard.md`. Includes SLO attainment, outstanding
  action items, and links to recent incidents.
- **Executive dashboard** – Tableau workbook `Reliability Overview` aggregates
  SLA/SLO status, customer impact, and burn-down charts for leadership reviews.
- **Audit trail** – All changes to SLO definitions and escalation policy are
  logged in the `reliability-changelog` Confluence space with links back to RFCs
  and postmortems.

Change proposals to SLO targets or escalation rules must be tracked via RFC with
sign-off from SRE lead, affected domain owner, and product counterpart.

## Appendix: Glossary

- **SLI** – Service Level Indicator; quantitative measurement such as success
  rate or latency.
- **SLO** – Service Level Objective; the target threshold an SLI must meet.
- **SLA** – Service Level Agreement; externally communicated reliability
  commitment.
- **Error budget** – Allowable unreliability `(1 - SLO target)` measured over
  the objective’s window.
- **Burn rate** – Forecast of how quickly the current error budget will be
  exhausted given recent performance.
