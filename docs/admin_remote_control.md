# Administrative Kill-Switch Remote Control

This guide explains how to enable and operate the secure administrative kill-switch endpoint shipped with TradePulse.

## Overview

TradePulse exposes a protected FastAPI surface at `/admin/kill-switch` that lets on-call operators inspect, engage, and reset the global `RiskManager` kill-switch. When the switch is engaged, all new orders are rejected to prevent further trading activity while the incident is being investigated.

Key properties:

- **Token-protected** – a shared secret token is required in every request (production deployments should upgrade to SSO or JWT).
- **Audited** – every action is recorded through the structured audit logger with a tamper-evident HMAC signature.
- **Idempotent** – repeated requests while the switch is active are acknowledged and logged as reaffirmations, and resets that find the switch already clear are logged as no-ops.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/admin/kill-switch` | Return the current kill-switch status without mutating it. |
| `POST` | `/admin/kill-switch` | Engage (or reaffirm) the kill-switch with an operator-provided reason. |
| `DELETE` | `/admin/kill-switch` | Reset the kill-switch; repeated invocations remain idempotent. |

The canonical OpenAPI schema for these operations lives at [`docs/api/admin_remote_control_openapi.yaml`](api/admin_remote_control_openapi.yaml).

## Configuration

Set the following environment variables before starting the FastAPI application:

| Variable | Description |
| --- | --- |
| `TRADEPULSE_ADMIN_TOKEN` | Static bearer token required in the `X-Admin-Token` header. |
| `TRADEPULSE_AUDIT_SECRET` | Secret used to sign audit records for integrity verification. |
| `TRADEPULSE_ADMIN_SUBJECT` | Default subject recorded for audit events when no `X-Admin-Subject` header is provided. |
| `TRADEPULSE_ADMIN_RATE_LIMIT_MAX_ATTEMPTS` | Maximum administrative attempts allowed within the rate-limit window (default `5`). |
| `TRADEPULSE_ADMIN_RATE_LIMIT_INTERVAL_SECONDS` | Rolling window in seconds for the administrative rate limiter (default `60`). |
| `TRADEPULSE_AUDIT_WEBHOOK_URL` | Optional HTTPS endpoint that receives a JSON copy of every administrative audit event. |

> **Important:** Development defaults are provided (`dev-admin-token`, `dev-audit-secret`) to simplify local testing. Always override them in production.

### Optional Headers

- `X-Admin-Subject`: Overrides the default administrator subject stored in audit logs. Use it to provide the operator's username or SSO principal.

## Request Flow

1. **Inspect state:**
   ```bash
   curl -H "X-Admin-Token: $TRADEPULSE_ADMIN_TOKEN" \
        -H "X-Admin-Subject: $(whoami)" \
        https://risk.tradepulse.example.com/admin/kill-switch
   ```
2. **Engage / reaffirm:**
   ```bash
   curl -X POST \
        -H "Content-Type: application/json" \
        -H "X-Admin-Token: $TRADEPULSE_ADMIN_TOKEN" \
        -H "X-Admin-Subject: $(whoami)" \
        -d '{"reason": "manual intervention after monitoring alert"}' \
        https://risk.tradepulse.example.com/admin/kill-switch
   ```
3. **Reset:**
   ```bash
   curl -X DELETE \
        -H "X-Admin-Token: $TRADEPULSE_ADMIN_TOKEN" \
        -H "X-Admin-Subject: $(whoami)" \
        https://risk.tradepulse.example.com/admin/kill-switch
   ```

## Responses

A successful engagement returns:

```json
{
  "status": "engaged",
  "kill_switch_engaged": true,
  "reason": "manual intervention after monitoring alert",
  "already_engaged": false
}
```

If the switch was previously active, `status` becomes `"already-engaged"` and `already_engaged` is `true`.

A reset that clears an active switch returns:

```json
{
  "status": "reset",
  "kill_switch_engaged": false,
  "reason": "manual intervention after monitoring alert",
  "already_engaged": true
}
```

If the switch is already clear the API responds with `status` `"already-clear"` while keeping `already_engaged` `false` to signal that no change was required. Reads return `"engaged"` or `"disengaged"` depending on the current state.

## Audit Logging

Audit events include the following fields:

- `event_type`: `kill_switch_state_viewed`, `kill_switch_engaged`, `kill_switch_reaffirmed`, `kill_switch_reset`, or `kill_switch_reset_noop`
- `actor`: Administrator subject (from `X-Admin-Subject` or default)
- `ip_address`: Remote IP extracted from the request
- `details`: Structured metadata containing the provided reason and whether the switch was already active
- `signature`: HMAC-SHA256 signature computed with `TRADEPULSE_AUDIT_SECRET`

Use `AuditLogger.verify(record)` to validate stored entries if tampering is suspected.

## Testing

Run the dedicated unit tests with:

```bash
pytest tests/admin/test_remote_control.py
```

The suite validates authentication, kill-switch semantics, and audit logging integrity.

## Production Recommendations

- Replace the static token with your organisation's SSO or JWT solution by extending `TokenAuthenticator`.
- Persist audit records to an append-only datastore (for example, write-through to SIEM or immutable storage).
- Configure alerts on the `kill_switch_reaffirmed` and `kill_switch_reset_noop` events to avoid unnoticed overrides or repeated ineffective resets.
- Integrate the kill-switch status into your incident response runbooks so that restarts include a reset step if appropriate, using the `DELETE /admin/kill-switch` endpoint for consistency with audit trails.
