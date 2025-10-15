# Deployment Guide

This guide outlines the production deployment requirements for TradePulse, including infrastructure, secret management, and operation of the live trading runner. It complements the [Production Cutover Readiness Checklist](../reports/prod_cutover_readiness_checklist.md).

## Infrastructure Requirements

| Component | Purpose | Recommended Baseline |
|-----------|---------|----------------------|
| **PostgreSQL 15+** | Stores strategy state, OMS snapshots, and execution audit trails. | Highly available cluster (e.g., managed Postgres, Patroni) with point-in-time recovery enabled. |
| **Kafka 3.5+** (or compatible message bus) | Distributes ticks, signals, and order events between ingestion, strategy, and execution services. | Three-broker cluster with replication factor ≥ 3, rack-aware placement, and topic-level ACLs. |
| **Prometheus + Alertmanager** | Scrapes metrics from TradePulse services, including the live trading loop heartbeat. | Dedicated metrics namespace with 15 s scrape interval and long-term storage via Thanos or Cortex. |
| **Object storage (S3/GCS/MinIO)** | Optional for historical data snapshots and strategy artifacts. | Versioned bucket with lifecycle policies to control retention. |
| **Secrets backend (Vault, AWS Secrets Manager, GCP Secret Manager)** | Centralised distribution of exchange credentials and API tokens. | Configure per-environment namespaces and audit logging. |

### Networking & Security

Refer to the [Production Security Architecture](security/architecture.md) for a
full description of the edge, DMZ, and core tiers, including the logging and
identity requirements that underpin these controls.

- Restrict inbound access to OMS and connector hosts using security groups or firewall rules.
- Enforce mTLS between strategy services and Kafka/Postgres where supported.
- Mirror Prometheus metrics to your SIEM for long-term incident investigations.

### Kafka Broker Security Configuration

- TradePulse expects Kafka clusters to expose TLS endpoints (`security_protocol` of `SSL` or `SASL_SSL`). Provide the CA bundle path via `EventBusConfig.ssl_cafile` and, when using mutual TLS, supply the signed client certificate and key files.
- Rotate broker and client certificates on a fixed cadence (e.g., quarterly). Deploy new files alongside the old ones, then restart services to reload credentials before revoking the previous certificates.
- When SASL is enabled, configure ACLs per topic and per consumer group. Bind the SASL principal used by TradePulse to the event topics defined in `core/messaging/event_bus.py` and deny wild-card access to minimise blast radius.
- Document the certificate and ACL owners in your runbooks so incident responders know who to contact when a rotation or ACL change is required.

## Secret Management Expectations

TradePulse loads sensitive credentials exclusively from environment variables or injected secret files.

1. **Source of truth** – Store live venue keys in your secret manager; do not commit them to Git.
2. **Rotation** – Align key rotation policies with venue requirements. The live trading loop supports hot credential reloads when files in the `state_dir` change.
3. **Distribution** – Inject credentials during deployment (e.g., Kubernetes secrets, HashiCorp Vault Agent) and expose them as environment variables such as `BINANCE_API_KEY` and `COINBASE_API_SECRET` as documented in [Configuration](configuration.md#exchange-connector-credentials).
4. **Audit** – Enable secret manager audit trails and configure alerts for unusual access patterns.

The administrative FastAPI surface consumes the `TRADEPULSE_AUDIT_SECRET` via a managed file watcher that honours rotations at runtime. When you mount `TRADEPULSE_AUDIT_SECRET_PATH` (and, optionally, `TRADEPULSE_SIEM_CLIENT_SECRET_PATH`) into the container, the service refreshes the keys according to `TRADEPULSE_SECRET_REFRESH_INTERVAL_SECONDS` without restarts. Ensure your secret manager agent keeps the files up to date and enforces length policies that satisfy the defaults (16+ characters for audit signatures).

## Configuring the Live Trading Runner

The live runner is implemented in [`execution/live_loop.py`](../execution/live_loop.py) and orchestrates connectors, the order management system, and risk controls.

1. **Prepare the state directory** – Mount a persistent volume for the runner and point `LiveLoopConfig.state_dir` to it. Each venue will receive a `${venue}_oms.json` snapshot used for reconciliation.
2. **Supply credentials** – Provide a mapping of venue names to API keys when creating `LiveLoopConfig(credentials=...)`. If you rely on environment variables only, leave the mapping empty and let connectors pull from the process environment.
3. **Bootstrap connectors** – Instantiate `ExecutionConnector` implementations for every venue you intend to trade. Ensure that WebSocket endpoints and REST base URLs are configured for the production environment.
4. **Risk management** – Construct a `RiskManager` with the relevant guards (position limits, P&L circuit breakers, kill switches). The live runner emits lifecycle events via `on_kill_switch`, `on_reconnect`, and `on_position_snapshot` signals—subscribe automation to these hooks for observability and fail-safes.
5. **Start the loop** – Call `LiveExecutionLoop.start(cold_start=False)` during deployment. Use `cold_start=True` only for the first cutover to avoid reusing stale state.
6. **Monitoring** – Scrape the metrics collector referenced by the loop (see `core.utils.metrics`). Ensure the following series are present: `live_loop_heartbeat`, `live_loop_orders_submitted_total`, and `live_loop_position_snapshot_timestamp`.
7. **Kill switch drills** – Schedule quarterly tests where the kill switch is triggered and verify that connectors disconnect, metrics report the event, and PagerDuty receives notifications.

## Configuration Promotion Workflow

1. Stage configuration changes in `configs/` and validate using paper trading or sandbox credentials.
2. Use the `reports/release_readiness.md` template to capture sign-off from risk and platform engineering.
3. For Kubernetes deployments, package configs as ConfigMaps with versioned labels; for VM-based deployments, store them in GitOps repositories and roll out via Ansible or Terraform.
4. Always run the regression test suite (`make test`) before promoting a build.

## Rollback Procedures

TradePulse rollbacks tie directly to the [Production Cutover Readiness Checklist](../reports/prod_cutover_readiness_checklist.md):

1. **Trigger conditions** – Monitor the SLO guardrails defined in the checklist (error rate > 2%, p95 latency > 500 ms, or metric gaps). When a breach occurs, the AutoRollbackGuard should emit the rollback callback.
2. **Execution** – Stop the live runner (`LiveExecutionLoop.shutdown()`), scale down new versions, and redeploy the previous tagged release from your artifact registry.
3. **Data reconciliation** – Restore OMS snapshots from the `state_dir` backups. Validate Postgres replicas and Kafka offsets against the pre-cutover baseline.
4. **Verification** – Re-run the checklist items marked as rollback drills to confirm the environment returned to nominal state.
5. **Postmortem** – File an incident report and attach telemetry exports (Prometheus, logs, Kafka lag metrics) for root-cause analysis.

## Troubleshooting Deployment Issues

- **Kafka consumer lag** – Inspect topic consumer groups and ensure partitions are balanced. If lag persists, scale execution workers horizontally.
- **Prometheus scrape failures** – Confirm service discovery labels include the live runner endpoints and TLS certificates are valid.
- **State reconciliation loops** – Remove corrupted OMS snapshots and restart with `cold_start=True`; the loop will regenerate clean state from the venues.
- **Credential mismatches** – Rotate secrets in the manager and restart pods to pick up the new values. Subscribe to the `on_kill_switch` signal to ensure trading halts during mismatches.

For additional operational policies, refer to [`docs/operational_readiness_runbooks.md`](operational_readiness_runbooks.md) and [`docs/incident_playbooks.md`](incident_playbooks.md).
