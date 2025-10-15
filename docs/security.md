# Secret Management and Rotation

TradePulse secures administrative interfaces and audit logging through a secret
management facade that wraps cloud providers such as AWS Secrets Manager or
HashiCorp Vault. Runtime components resolve secret identifiers at startup and
refresh cached credentials before they expire. This document describes how to
configure the integration, rotate credentials safely, and observe secret access
health.

## Configuration Overview

Secrets are referenced by stable identifiers. In development the identifiers map
directly to environment variables, while production deployments should resolve
the same identifiers to managed secret store paths.

| Setting                               | Description                                                       |
|---------------------------------------|-------------------------------------------------------------------|
| `TRADEPULSE_AUDIT_SECRET_ID`          | Secret manager identifier for the audit logging signing key.      |
| `TRADEPULSE_ADMIN_TOKEN_ID`           | Identifier for the administrative bearer token (optional).        |
| `TRADEPULSE_AUDIT_SECRET`             | Development fallback value when using the in-process env provider.|
| `TRADEPULSE_ADMIN_TOKEN`              | Development fallback value for the admin token.                   |
| `<ID>_TTL_SECONDS`                    | Optional override for the cached lifetime of a secret.            |

The FastAPI service requests both identifiers during startup. If a secret cannot
be resolved a descriptive runtime error is raised referencing the missing ID.

## Rotation Runbook

1. **Prepare the new secret version** in your secret manager (for example, write
a new AWSSM version under `TRADEPULSE_AUDIT_SECRET_ID`). Ensure the new value
satisfies the complexity policy—at least 16 characters including a mix of upper
case, lower case, digits, and symbols.
2. **Update the metadata** in the secret manager to set an appropriate TTL or
expiration window. The default refresh margin is five minutes; choose a TTL
significantly longer than this window.
3. **Deploy the change**. Restart the service or trigger a configuration reload
so that the global secret manager registers the updated identifier. The
`AuditLogger` and `TokenAuthenticator` subscribe to rotation events and update
in-memory keys automatically.
4. **Verify rotation**. Check application logs for the `audit.secret.loaded` and
`admin.token.rotated` events. Metrics emitted through the secret manager
(`secret.access` and `secret.near_expiry`) can be scraped by your monitoring
pipeline to confirm healthy refresh behaviour.
5. **Retire the previous version** once verification succeeds. Historical audit
records remain verifiable because the logger caches keys by version during the
process lifetime.

## Monitoring and Alerting

The secret manager emits structured logs and optional metrics for every access:

* `secret.access` — informational log/metric documenting which secret was read.
* `secret.near_expiry` — warning emitted when a secret approaches its expiry
  window. Near-expiry events should trigger proactive rotation workflows.
* `secret.rotated` — indicates that a fresh value has been fetched and cached.

Administrators should scrape these events into their observability stack and
configure alerts for frequent `secret.near_expiry` warnings or failed rotation
callbacks.

## Policy-as-Code Enforcement

During test execution `pytest_sessionstart` enforces baseline policies defined in
`conftest.py`:

* Required identifiers must be present: `TRADEPULSE_AUDIT_SECRET_ID` and
  `TRADEPULSE_ADMIN_TOKEN_ID`.
* Secrets must meet the complexity heuristic described above.
* Remaining TTL must exceed the policy threshold (one hour for audit secrets,
  thirty minutes for administrative tokens).

Violations fail the test run early, providing actionable remediation messages.

## Operational Tips

* Always populate both the identifier (e.g. `TRADEPULSE_AUDIT_SECRET_ID`) and a
  store-specific value. In production this typically means provisioning a new
  version in your secret manager and updating the application deployment
  configuration.
* Use the built-in metrics to drive dashboards that highlight rotation cadence
  and the age of cached credentials.
* When running locally you can override secrets by exporting environment
  variables with strong placeholder values that satisfy the complexity policy.
