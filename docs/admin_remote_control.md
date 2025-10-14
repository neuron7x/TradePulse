# Administrative Kill-Switch Remote Control

This guide explains how to enable and operate the secure administrative kill-switch endpoint shipped with TradePulse.

## Overview

TradePulse exposes a protected FastAPI endpoint at `/admin/kill-switch` that toggles the global `RiskManager` kill-switch. When the switch is engaged, all new orders are rejected to prevent further trading activity while the incident is being investigated.

Key properties:

- **Token-protected** – a shared secret token is required in every request (production deployments should upgrade to SSO or JWT).
- **Audited** – every action is recorded through the structured audit logger with a tamper-evident HMAC signature.
- **Idempotent** – repeated requests while the switch is active are acknowledged and logged as reaffirmations.

## Configuration

Set the following environment variables before starting the FastAPI application:

| Variable | Description |
| --- | --- |
| `TRADEPULSE_ADMIN_TOKEN` | Static bearer token required in the `X-Admin-Token` header. |
| `TRADEPULSE_AUDIT_SECRET` | Secret used to sign audit records for integrity verification. |
| `TRADEPULSE_ADMIN_ALLOW_LIST` | Optional comma-separated list of CIDR blocks permitted to call the endpoint (e.g. `10.0.0.0/8,2001:db8::/32`). |
| `TRADEPULSE_ADMIN_SUBJECT_RATE_LIMIT` | Sliding window rate limit applied per subject in the format `attempts/windowSeconds` (default `5/60`). |
| `TRADEPULSE_ADMIN_IP_RATE_LIMIT` | Sliding window rate limit applied per source IP, formatted as `attempts/windowSeconds` (default `30/60`). |

> **Important:** Development defaults are provided (`dev-admin-token`, `dev-audit-secret`) to simplify local testing. Always override them in production.

### Optional Headers

- `X-Admin-Subject`: Overrides the default administrator subject stored in audit logs. Use it to provide the operator's username or SSO principal.

## Request Flow

1. Issue a `POST` request to `/admin/kill-switch` with a JSON body:
   ```json
   {
     "reason": "manual intervention after monitoring alert"
   }
   ```
2. Include the `X-Admin-Token` header containing the configured administrative token.
3. Optionally include `X-Admin-Subject` to capture the operator identity in the audit log.

## Responses

A successful invocation returns:

```json
{
  "status": "engaged",
  "kill_switch_engaged": true,
  "reason": "manual intervention after monitoring alert",
  "already_engaged": false
}
```

If the switch was previously active, `status` becomes `"already-engaged"` and `already_engaged` is `true`.

## Audit Logging & Rate Limiting

Audit events include the following fields:

- `event_type`: `kill_switch_engaged`, `kill_switch_reaffirmed`, `kill_switch_rate_limited`, or `kill_switch_access_denied`
- `actor`: Administrator subject (from `X-Admin-Subject` or default)
- `ip_address`: Remote IP extracted from the request
- `details`: Structured metadata containing the provided reason, whether the switch was already active, and—in the case of denials—the calculated retry delays or enforced allow-lists.
- `signature`: HMAC-SHA256 signature computed with `TRADEPULSE_AUDIT_SECRET`

Rate-limit denials (`HTTP 429`) include the subject/IP retry-after durations in `details`. CIDR allow-list denials (`HTTP 403`) capture the permitted CIDR entries for forensic review. Use `AuditLogger.verify(record)` to validate stored entries if tampering is suspected.

### Hydra Override Example

When using Hydra to manage application configuration, provide an override block similar to the following to tailor access controls:

```yaml
admin_access:
  allow_cidrs:
    - "203.0.113.0/24"
    - "2001:db8::/48"
  subject_rate_limit:
    max_attempts: 3
    window_seconds: 60
  ip_rate_limit:
    max_attempts: 10
    window_seconds: 60
```

This mirrors the structure expected by `AdminAccessPolicyConfig` (passed to `create_app(admin_access_config=...)`).

### Prometheus Metrics

The following Prometheus series are emitted through the standard exporters:

- `tradepulse_admin_remote_control_requests_total{outcome="success|denied|throttled"}` – counts of administrative attempts.
- `tradepulse_admin_remote_control_throttle_seconds` – histogram of Retry-After delays advertised to throttled clients.

## Testing

Run the dedicated unit tests with:

```bash
pytest tests/admin/test_remote_control.py
```

The suite validates authentication, kill-switch semantics, and audit logging integrity.

## Production Recommendations

- Replace the static token with your organisation's SSO or JWT solution by extending `TokenAuthenticator`.
- Persist audit records to an append-only datastore (for example, write-through to SIEM or immutable storage).
- Configure alerts on the `kill_switch_reaffirmed` event to avoid unnoticed overrides.
- Integrate the kill-switch status into your incident response runbooks so that restarts include a reset step if appropriate.
