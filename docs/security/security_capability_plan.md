# TradePulse Security Capability Development Plan

## Purpose
This plan inventories the primary decision centres across TradePulse and outlines the security maturation roadmap needed to reach a best-in-class posture. It synthesises code-level controls, architecture documentation, and governance policies to deliver actionable, sequenced upgrades.

## Decision Centre Assessment

### 1. Market Data Ingestion & Schema Governance
- **Current posture**
  - Connectors enforce Avro schema contracts at runtime, use dead-letter queues, and backoff logic to contain malformed market data and network faults.【F:core/data/connectors/market.py†L1-L188】
  - Governance policies already require schema catalogues and lineage for raw sources, with zero-trust mesh restrictions between ingestion and downstream services.【F:docs/governance.md†L1-L157】
- **Risks**
  - Lack of automated provenance attestation for ingestion binaries and configuration bundles.
  - Dead-letter drains are local-only; without forwarding, repeated anomalies could hide systemic tampering.
- **Planned improvements**
  1. Sign ingestion container images and configuration bundles via sigstore, verifying signatures in deployment controllers before rollout (Q1).
  2. Route dead-letter queue drains into the governance topic with schema metadata, enabling correlation with access logs (Q1).
  3. Expand connector health gating to require attested network zones and secret rotation evidence prior to enabling new venues (Q2).

### 2. Feature Store Integrity & Data Contracts
- **Current posture**
  - Online and offline feature stores enforce TTL-aware retention, column compatibility checks, and cryptographic hash parity using SHA-256 with constant-time comparisons.【F:core/data/feature_store.py†L1-L568】
  - Governance mandates semantic versioning, change management, and privacy tagging for all data contracts.【F:docs/governance.md†L58-L157】
- **Risks**
  - Integrity reports require manual inspection; no automated quarantine of divergent feature views.
  - Absence of differential privacy or noise injection for analyst exports increases risk of sensitive data leakage from aggregated tiers.
- **Planned improvements**
  1. Implement automated quarantine workflow: failing integrity checks trigger feature view disablement and on-call paging, with self-healing upon remediation (Q1).
  2. Add contract-as-code validation (OPA/Rego) for retention/PII policies directly in feature store sync jobs (Q2).
  3. Introduce privacy budget accounting for analytics exports, with noise calibration approved by compliance (Q3).

### 3. Domain & Application Layer Controls
- **Current posture**
  - Domain primitives enforce strict validation (e.g., order lifecycle, signal confidence bounds) ensuring upstream systems cannot inject malformed trading directives.【F:domain/order.py†L1-L167】【F:domain/signal.py†L1-L73】
  - Architecture mandates UI and infrastructure isolation from domain packages to contain blast radius.【F:docs/ARCHITECTURE.md†L23-L39】
- **Risks**
  - No formal threat model for cross-layer abuse (e.g., UI-sourced DTOs bypassing domain constructors).
  - Limited runtime policy enforcement (RBAC, context-aware approval) at the application services level.
- **Planned improvements**
  1. Run STRIDE-based threat modelling for domain-service boundaries; convert findings into automated security tests (Q1).
  2. Embed attribute-based access control (ABAC) middleware in application APIs, aligning with governance role definitions (Q2).
  3. Add secure coding checklists and fuzz tests for DTO mappers to ensure resilience against tampered payloads (Q2).

### 4. Execution & Risk Orchestration
- **Current posture**
  - Live execution loop coordinates connectors, OMS, and risk management with kill-switch hooks and credential scoping via configuration.【F:execution/live_loop.py†L1-L200】
  - Governance enforces dual control for kill switches and runtime approvals for strategy activation.【F:docs/governance.md†L89-L157】
- **Risks**
  - Credential handling relies on filesystem paths; no hardware-backed secrets or attested sessions for high-value venues.
  - Kill-switch signalling lacks cryptographic authentication, enabling potential spoofing on compromised hosts.
- **Planned improvements**
  1. Integrate hardware security module (HSM) backed signing for OMS credentials and session establishment (Q1).
  2. Wrap kill-switch and reconnect signals in signed/verifiable events with rotation of signing keys managed by Vault (Q2).
  3. Deploy continuous controls monitoring to verify OMS/risk configs against signed baselines every 15 minutes (Q2).

### 5. Analytics, UI, and Delivery Channels
- **Current posture**
  - Architecture roadmap targets a gRPC-web gateway feeding the Next.js dashboard, which must adhere to protobuf contracts.【F:docs/ARCHITECTURE.md†L1-L21】
  - Governance restricts UI analyst roles to curated datasets with row/column security enforced through OIDC groups and reverse-proxy ACLs.【F:docs/governance.md†L8-L41】
- **Risks**
  - Pending gRPC-web gateway introduces potential CSRF/session fixation risks without explicit mitigations.
  - Frontend telemetry is not yet tied into security analytics for anomaly detection.
- **Planned improvements**
  1. Implement mutual TLS between dashboard and gateway, with short-lived SPA tokens and nonce-based CSRF protection (Q1).
  2. Add browser telemetry beacons emitting signed session events to the governance topic for behavioural analytics (Q2).
  3. Enforce content security policy (CSP) and security header linting in CI/CD before dashboard deployments (Q1).

### 6. Observability, Compliance, and Governance Backbone
- **Current posture**
  - Governance documentation mandates sigstore signing, immutable audit snapshots, and zero-trust mesh communication, establishing strong baseline controls.【F:docs/governance.md†L89-L157】
  - Security policy defines responsible disclosure and dependency auditing workflows.【F:SECURITY.md†L1-L120】
- **Risks**
  - Control evidence is described but not aggregated into a unified risk dashboard or automated compliance attestation.
  - Penetration testing cadence exists, but red-team findings are not explicitly linked to backlog automation.
- **Planned improvements**
  1. Build a continuous assurance dashboard aggregating governance events, SBOM findings, and integrity reports with risk scoring (Q1).
  2. Automate remediation ticket creation from dependency audits and pen-test findings, tracking mean-time-to-remediate (Q1).
  3. Establish purple-team exercises aligning defensive detections with kill-chain stages and feeding detection-as-code repositories (Q3).

## Roadmap & Milestones
| Quarter | Theme | Key Deliverables |
| --- | --- | --- |
| Q1 | Foundational hardening | Sigstore enforcement for ingestion; automated feature view quarantine; STRIDE threat model; HSM-backed OMS secrets; CSP & CSRF controls; assurance dashboard. |
| Q2 | Policy automation | ABAC middleware; connector health attestation; Rego-based contract enforcement; signed kill-switch events; telemetry beacons; automated remediation tickets. |
| Q3 | Advanced resilience | Privacy budgeting; purple-team programme; continuous controls monitoring; analytics export safeguards. |
| Q4 | Optimisation & certification | Prepare for SOC 2 Type II / ISO 27001 audits, leveraging accumulated evidence and dashboards; expand automation coverage to new asset classes. |

## Execution Governance
- **Security Steering Committee**: chaired by CISO, includes leads from data, execution, and platform teams. Reviews progress monthly, aligning with governance council requirements.【F:docs/governance.md†L1-L157】
- **Control Owners**: module leads (ingestion, feature store, execution, UI) accountable for implementing roadmap items with security engineering support.
- **Measurement**: track mean-time-to-detect (MTTD), mean-time-to-remediate (MTTR), control coverage (% of connectors signed, % feature views monitored), and compliance SLAs (audit evidence freshness).

## Next Steps
1. Ratify this plan at the next governance council meeting and publish commitments in the internal change calendar.
2. Create JIRA epics per roadmap theme with control owners and acceptance criteria referencing this document.
3. Schedule quarterly reviews to recalibrate based on threat landscape shifts and post-incident learnings.
