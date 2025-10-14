# Administrative Kill-Switch Remote Control

This guide explains how to enable and operate the secure administrative kill-switch endpoint shipped with TradePulse.

## Overview

TradePulse exposes a protected FastAPI endpoint at `/admin/kill-switch` that toggles the global `RiskManager` kill-switch. When the switch is engaged, all new orders are rejected to prevent further trading activity while the incident is being investigated.

Key properties:

- **Short-lived credentials** – HMAC-backed bearer tokens issued by the platform secret manager and bound to specific scopes and audiences.
- **mTLS enforcement** – every request must present the expected client certificate thumbprint to pass verification.
- **Audited** – every action is recorded through the structured audit logger with a tamper-evident HMAC signature.
- **Idempotent** – repeated requests while the switch is active are acknowledged and logged as reaffirmations.

## Configuration

Set the following environment variables before starting the FastAPI application:

| Variable | Description |
| --- | --- |
| `TRADEPULSE_ADMIN_SIGNING_KEY` | HMAC signing key retrieved from the secret manager for issuing admin credentials. |
| `TRADEPULSE_AUDIT_SECRET` | Secret used to sign audit records for integrity verification. |

> **Important:** Development defaults are provided (`dev-admin-signing-key`, `dev-audit-secret`) to simplify local testing. Always override them in production.

## Request Flow

1. Issue a `POST` request to `/admin/kill-switch` with a JSON body:
   ```json
   {
     "reason": "manual intervention after monitoring alert"
   }
   ```
2. Include the `Authorization: Bearer <token>` header where `<token>` is the short-lived credential minted from the secret manager.
3. Provide `X-Client-Cert-Thumbprint` with the SHA-256 thumbprint of the authenticated mTLS client certificate.

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

## Audit Logging

Audit events include the following fields:

- `event_type`: `kill_switch_engaged` or `kill_switch_reaffirmed`
- `actor`: Administrator subject extracted from the credential payload
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

- Integrate the secret manager minting workflow with your SSO provider so that `ShortLivedTokenVerifier` receives signed JWTs or HMAC tokens tied to operator identities.
- Persist audit records to an append-only datastore (for example, write-through to SIEM or immutable storage).
- Configure alerts on the `kill_switch_reaffirmed` event to avoid unnoticed overrides.
- Integrate the kill-switch status into your incident response runbooks so that restarts include a reset step if appropriate.
