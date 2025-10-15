---
title: Identity and Access Management
description: Centralised identity architecture covering SSO, MFA, least-privilege policy automation, and RBAC enforcement for TradePulse.
---

# Identity and Access Management

TradePulse relies on a layered IAM programme that federates workforce and service identities through a single control plane. The objective is to ensure that every interactive session and automation workflow is authenticated via single sign-on (SSO), verified with multi-factor authentication (MFA), and authorised under least-privilege, role-based access control (RBAC) policies.

## Identity Federation and SSO

- **Primary IdP** – Azure AD, Okta, or another OIDC-compliant provider acts as the source of truth for workforce identities. All applications integrate through the shared `tradepulse` tenant/app registration.
- **SSO enablement** – Users authenticate through the IdP portal. OIDC/OAuth clients consume signed JWTs that include the `sub`, `email`, `groups`, and `assurance_level` claims required by the control plane.
- **Session brokers** – The DMZ identity proxy exchanges external IdP tokens for internal audience JWTs referenced by `TRADEPULSE_OAUTH2_AUDIENCE`. Sidecars enforce token validation before requests reach FastAPI services.
- **Break-glass accounts** – Stored in a hardware security module (HSM)-backed vault with manual rotation procedures. Break-glass access is logged and requires post-incident review.

### Implementation Checklist

1. Register TradePulse as a confidential OIDC client with redirect URIs for the CLI, UI, and admin tools.
2. Configure the IdP to issue group membership in the `groups` claim and assurance level (reflecting MFA status) in the `acr` or custom claim.
3. Deploy the identity proxy with JWKS caching and strict issuer/audience validation.
4. Validate SSO end-to-end by exercising login flows for CLI, UI, and API clients.

## Multi-Factor Authentication (MFA)

- **Workforce enforcement** – Conditional access policies in the IdP require MFA for every interactive session, with step-up prompts triggered for privileged role elevation or sensitive operations (e.g., portfolio deployments).
- **Authenticator coverage** – Support security keys (WebAuthn/FIDO2) as the primary factor, with TOTP apps as the fallback. SMS/voice should be disabled except for break-glass scenarios.
- **Service accounts** – Non-interactive workloads use workload identity federation (OIDC trust relationships). Secrets-based service accounts are disallowed unless covered by rotation automation and explicit exceptions.
- **Monitoring** – Stream MFA challenge results to `observability/audit/iam_events.jsonl` for correlation with application audit trails.

### Implementation Checklist

1. Enforce IdP policies requiring registered security keys or TOTP apps before granting application access.
2. Configure risk-based policies to trigger re-authentication when IP reputation or device posture deviates from baselines.
3. Integrate SIEM alerts for repeated MFA failures or bypass attempts.
4. Document emergency MFA bypass in the runbooks with explicit approval workflow.

## Least-Privilege RBAC Model

RBAC scopes map to TradePulse capabilities across portfolio management, model operations, and platform administration. Roles are defined declaratively in the IAM configuration repository and rendered into OPA bundles for runtime enforcement.

| Role | Scope | Example Permissions |
| --- | --- | --- |
| `viewer` | Read-only access to analytics dashboards and portfolio state. | `analytics:read`, `portfolio:read` |
| `quant` | Build, backtest, and stage strategies. No production deployment rights. | `strategy:create`, `backtest:run`, `model:register` |
| `trader` | Execute approved strategies in production windows. | `execution:launch`, `execution:pause`, `portfolio:adjust` |
| `mlops` | Promote models, manage feature store pipelines, oversee model registry. | `model:promote`, `featurestore:sync`, `artifact:approve` |
| `sre` | Operate platform infrastructure, manage incident response. | `deployment:rollout`, `system:configure`, `audit:view` |
| `admin` | Restricted to IAM/security team for policy updates and break-glass access. | `iam:modify`, `role:assign`, `secret:rotate` |

### Policy Authoring Lifecycle

1. **Source control** – RBAC definitions live in `configs/iam/roles.yaml` and `configs/iam/policies.yaml`. Every change is reviewed via pull requests with approval from role owners.
2. **Validation** – CI pipelines run `opa test` and static checks to ensure policies compile and meet regression expectations.
3. **Bundle distribution** – Approved policies are packaged into OPA bundles and distributed via the internal artifact registry. Deployments subscribe to bundle updates through sidecar bootstrap settings.
4. **Runtime enforcement** – FastAPI and control-plane services query the sidecar for allow/deny decisions, logging the decision ID for traceability.

### Access Request Workflow

- **Self-service portal** – Requests originate in the ITSM portal, capturing business justification, requested role, and expiration.
- **Approval chain** – Requires requester manager + system owner approval. Elevated roles (`sre`, `admin`) additionally require security sign-off.
- **Just-in-time elevation** – Short-lived grants issued via Terraform Cloud runs that modify `configs/iam/role_bindings.yaml` with expiry metadata.
- **Certification** – Quarterly reviews reconcile active bindings against HR rosters. Orphaned accounts or stale bindings trigger automatic revocation pull requests.

## Least-Privilege Guardrails

- **Segregation of duties** – No single user holds both `quant` and `trader` roles permanently. Exceptions need risk approval and expire within seven days.
- **Scoped secrets** – Secret scopes mirror RBAC permissions. The vault broker issues dynamic credentials tagged with the requestor identity and role context.
- **Environment isolation** – Non-production roles map to separate namespaces (`dev`, `staging`) preventing privilege creep into production clusters.
- **Observability** – Audit trails from FastAPI (`observability/audit/api_events.jsonl`) and IAM decisions (`observability/audit/iam_events.jsonl`) are correlated in the SIEM, enabling blast radius analysis and compliance reporting.

## Operational Runbooks

- **Onboarding** – Reference `docs/onboarding.md` for baseline role assignments (`viewer`, `quant`). MFA registration must be completed before any role grant.
- **Offboarding** – HR events trigger immediate revocation of SSO access, removal from role bindings, and secret invalidation. Terraform automation submits removal pull requests automatically.
- **Break-glass** – Documented in `docs/security/zero_trust_runbook.md`. All break-glass access is temporary, logged, and reviewed within 24 hours.
- **Incident response** – IAM anomalies escalate via the incident playbooks (`docs/incident_playbooks.md`). Emergency role freezes can be applied by the `admin` team by revoking bundle distribution.

## Metrics and Continuous Improvement

Track the following metrics to evaluate IAM health:

- Percentage of interactive sessions completed with hardware-based MFA factors.
- Mean time to approve or reject access requests.
- Count of policy violations blocked by OPA per environment.
- Number of orphaned identities detected during quarterly certifications.
- Coverage of service accounts migrated to workload identity federation versus static secrets.

Regularly review these metrics with the security steering group and incorporate lessons learned into the IAM roadmap and ADRs.
