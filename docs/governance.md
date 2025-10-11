# Governance and Data Controls

This guide defines the governance guardrails for TradePulse across access management, data contracts, privacy, and cataloguing. Policies apply to all production and pre-production environments unless explicitly waived by the Governance Council.

## Role-Based Access Control (RBAC)

| Role | Scope | Permitted Actions | Enforced Through |
| ---- | ----- | ----------------- | ---------------- |
| **Ingestion Operator** | Streaming and batch ingestion services | Manage connectors, configure schemas, run validation pipelines, rotate ingestion secrets. No direct database writes outside staging schemas. | IAM policies on `ingestion-*` services, service mesh policy allowing `POST/PUT` to ingestion APIs, GitOps repository permissions for connector configs. |
| **Backtest Engineer** | Research compute clusters and artifact stores | Launch backtest jobs, read historical market data, publish backtest reports, submit feature proposals. No production execution permissions. | RBAC in workflow orchestrator (e.g., Argo), read-only S3 bucket policies, GitHub team `backtest-dev`. |
| **Execution Trader** | Live execution and risk control services | Approve deployment of execution models, adjust risk limits, halt strategies, monitor execution telemetry. Cannot modify ingestion connectors or raw data. | Fine-grained roles in execution control plane, feature-flag management, emergency break-glass tokens logged in PAM. |
| **UI Analyst** | Web UI and analytics workspaces | View aggregated dashboards, download approved reports, annotate anomalies. No direct access to raw feature stores or execution toggles. | OIDC group `ui-analyst`, reverse-proxy ACLs, row/column level security on BI datasets. |

### Service-Level Access Policies

1. **Zero trust mesh** – mTLS enforced between services with SPIFFE identities, limiting service-to-service calls to declared intents (e.g., ingestion services cannot call execution write endpoints).
2. **Environment segregation** – staging and production namespaces enforce namespace-level network policies; execution services only accept traffic from approved front doors.
3. **Secrets governance** – HashiCorp Vault policies scoped per role; dynamic secrets for databases expire within 1 hour; audited secret rotation via CI workflows.
4. **Least privilege automation** – Terraform modules expose role bindings as code with review requirements and automated drift detection (weekly `terraform plan` reports).

## Data Contracts

### Contract Types

| Data Tier | Owner | Schema & Quality Guarantees | Distribution | Consumers |
| --------- | ----- | -------------------------- | ------------ | --------- |
| **Raw** | Data Engineering | Immutable schema with additive fields, partitioned by ingest timestamp, mandatory lineage tags. Quality checks: format validation, checksum verification, source completeness threshold ≥95%. | Object storage (`raw/` prefix), replayable change data capture streams. | Feature pipelines, archive services. |
| **Aggregated** | Analytics Engineering | Derived tables with documented grain, aggregation windows, and null-handling rules. Quality checks: aggregation parity tests, anomaly detection (<3σ). | Warehouse schemas (`analytics.`), curated API endpoints. | UI dashboards, risk analytics, reporting. |
| **Feature** | ML Engineering | Feature store entities with versioned feature views, point-in-time correctness, and training-serving skew monitors. Quality checks: feature drift alerts, data freshness SLA <15 minutes. | Feature store registry, batch exports for model training. | Backtest platform, execution scoring services. |

### Change Management

- **Non-breaking changes** – additive columns, extended enumerations with defaults, new optional attributes. Require:
  - 3-day notice via `#data-contracts` channel.
  - Updated contract YAML in `data/contracts/` with semantic version patch increment.
  - Automated schema compatibility tests passing in CI.
- **Breaking changes** – column removal/rename, data type narrowing, primary key changes, SLA relaxations. Require:
  - 14-day RFC with impact analysis and rollback plan.
  - Approval from Data Governance Council and affected service owners.
  - Coordinated release window with feature flag or dual-write strategy.
  - Major version bump and migration playbook stored in `docs/migrations/`.

## Privacy and PII Handling

1. **Collection Policy** – Collect only fields required for trading compliance, regulatory reporting, or customer deliverables. All PII attributes must be catalogued with data owner sign-off and tagged in the metadata store.
2. **Masking & Tokenisation** – Apply irreversible hashing for persistent storage of identifiers; use format-preserving tokenisation for operational workflows. Access to de-tokenisation services requires break-glass approval with session recording.
3. **Retention & Deletion** – Raw PII limited to 90 days unless regulatory retention requires longer. Aggregated datasets must strip direct identifiers. Quarterly retention audits verify deletion jobs succeeded.
4. **CI/CD Enforcement** –
   - Static checks ensuring migrations referencing PII tables include masking functions.
   - Unit tests verifying PII columns are excluded from public exports.
   - Automated policy-as-code (OPA/Rego) gate in CI to block deployments if datasets lack privacy tags or retention rules.

## Data Catalog, Lineage, and Source Inventory

- **Central Metadata Store** – Use an OpenMetadata deployment storing dataset schemas, owners, SLAs, and privacy classifications. Each dataset entry includes contact information and runbooks.
- **Lineage Tracking** – Instrument ingestion and transformation jobs to emit OpenLineage events. Visual lineage graphs connect raw sources → aggregated tables → feature views → downstream services.
- **Versioning** – Datasets carry semantic versions aligned with data contract versions. Historical snapshots are persisted to enable point-in-time recovery and auditability.
- **Source Inventory** – Maintain an authoritative inventory (`data/sources.yaml`) listing external vendors, regulatory feeds, internal microservices, refresh cadence, and contractual obligations. Inventory updates trigger governance review through the change management workflow.
- **Access Transparency** – Usage analytics dashboards monitor dataset reads by role, surfacing anomalous access patterns for investigation.

## Operational Cadence

- Monthly governance review to assess RBAC exceptions, contract changes, and privacy incidents.
- Quarterly penetration test focused on data exfiltration paths and identity boundary hardening.
- Annual recertification of all data sources, with lineage validation and contract renewal.

