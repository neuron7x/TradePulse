# Administrative Kill-Switch Remote Control

This guide explains how to enable and operate the secure administrative kill-switch endpoint shipped with TradePulse.

## Overview

TradePulse exposes a protected FastAPI endpoint at `/admin/kill-switch` that toggles the global `RiskManager` kill-switch. When the switch is engaged, all new orders are rejected to prevent further trading activity while the incident is being investigated.

Key properties introduced in this hardening pass:

- **Token + Role authentication** – `X-Admin-Token` is validated with constant-time comparison and every administrator must present an `X-Admin-Roles` header that includes either `admin:kill-switch` or `admin:super`.
- **Rate limiting & idempotency** – a sliding-window limiter protects against abuse while mandatory `X-Idempotency-Key` headers guarantee that retried requests do not re-trigger the kill-switch.
- **Structured observability** – correlation IDs, Prometheus metrics, and OpenTelemetry traces are emitted for every invocation.
- **Atomic execution** – retries are guarded by exponential backoff, per-key locks, and monotonic-clock timeouts so that the risk manager is called exactly once per idempotent key.
- **Cache-aware** – recent kill-switch states are cached briefly to avoid unnecessary synchronous calls while remaining multi-process friendly.

## Configuration

Set the following environment variables before starting the FastAPI application:

| Variable | Description |
| --- | --- |
| `TRADEPULSE_ADMIN_TOKEN` | Static bearer token required in the `X-Admin-Token` header. |
| `TRADEPULSE_AUDIT_SECRET` | Secret used to sign audit records for integrity verification. |
| `TRADEPULSE_ADMIN_RATE_LIMIT` | (Optional) Override of requests per minute, e.g. `5`. |
| `TRADEPULSE_ADMIN_RATE_PERIOD` | (Optional) Override of the sliding window duration in seconds. |

> **Important:** Development defaults are provided (`dev-admin-token`, `dev-audit-secret`) to simplify local testing. Always override them in production.

### Optional Headers

- `X-Admin-Subject`: Overrides the default administrator subject stored in audit logs. Use it to provide the operator's username or SSO principal.
- `X-Correlation-ID`: If omitted, the API will generate a UUIDv4 value and propagate it to logs, metrics, and traces.

## Request Flow

1. Issue a `POST` request to `/admin/kill-switch` with a JSON body:
   ```json
   {
     "reason": "manual intervention after monitoring alert"
   }
   ```
2. Include the following headers:
   - `X-Admin-Token`: Configured secret token.
   - `X-Admin-Roles`: Comma-separated list containing `admin:kill-switch` or `admin:super`.
   - `X-Idempotency-Key`: Unique key per logical activation (UUIDs are recommended).
   - `X-Correlation-ID`: Optional value propagated into structured logs.
   - `X-Admin-Subject`: Optional override of the audit identity.
3. The handler validates the payload using Pydantic, checks authorization, applies the rate limiter, and executes the command with retries (exponential backoff and per-attempt timeout).

## Responses

A successful invocation returns:

```json
{
  "status": "engaged",
  "kill_switch_engaged": true,
  "reason": "manual intervention after monitoring alert",
  "already_engaged": false,
  "correlation_id": "f29a4c62-d6fd-47d5-9b43-95c5f49ad6b9",
  "idempotency_key": "3e1a9e4e-2962-4aa0-92ff-69fe80f74d23",
  "timestamp": "2024-07-04T12:41:33.082194+00:00"
}
```

If the switch was previously active, `status` becomes `"already-engaged"` and `already_engaged` is `true`.

### Error responses

- `403 Forbidden` – administrator authenticated but lacks the required role.
- `429 Too Many Requests` – rate limit exceeded for the subject within the configured window.
- `400 Bad Request` – validation failure (missing headers, malformed JSON, reason shorter than three characters).
- `503 Service Unavailable` – retries exhausted while calling the risk manager.
- `500 Internal Server Error` – unexpected failure bubbled up after sanitised logging.

## Audit Logging

Audit events include the following fields:

- `event_type`: `kill_switch_engaged` or `kill_switch_reaffirmed`
- `actor`: Administrator subject (from `X-Admin-Subject` or default)
- `ip_address`: Remote IP extracted from the request
- `details`: Structured metadata containing the provided reason, idempotency key, correlation ID, UTC timestamp, and whether the switch was already active
- `signature`: HMAC-SHA256 signature computed with `TRADEPULSE_AUDIT_SECRET`

Use `AuditLogger.verify(record)` to validate stored entries if tampering is suspected.

## Observability

- **Structured logs** – emitted via `core.utils.logging.StructuredLogger` with the correlation ID stored in `request.state.correlation_id` for middleware propagation.
- **Metrics** – Prometheus counters/histogram:
  - `tradepulse_admin_kill_switch_total`
  - `tradepulse_admin_kill_switch_rate_limited_total`
  - `tradepulse_admin_kill_switch_errors_total`
  - `tradepulse_admin_kill_switch_latency_seconds`
- **Tracing** – if OpenTelemetry is installed, the handler creates an `admin.kill_switch` span annotated with the administrator subject and correlation ID.

Hook these signals into your existing observability pipeline to detect abuse patterns and recovery timelines.

## Testing

Run the dedicated unit tests with:

```bash
pytest tests/unit/admin/test_remote_control_service.py \
       tests/integration/admin/test_remote_control_integration.py \
       tests/fuzz/admin/test_remote_control_fuzz.py \
       tests/property/admin/test_remote_control_properties.py
```

The suite validates authentication, kill-switch semantics, and audit logging integrity.

## Production Recommendations

- Replace the static token with your organisation's SSO or JWT solution by extending `TokenAuthenticator`.
- Persist audit records to an append-only datastore (for example, write-through to SIEM or immutable storage).
- Configure alerts on the `kill_switch_reaffirmed` event to avoid unnoticed overrides.
- Integrate the kill-switch status into your incident response runbooks so that restarts include a reset step if appropriate.
- For multi-process deployments, provide a distributed rate limiter and idempotency backend (Redis, DynamoDB, or PostgreSQL) via dependency injection to share state safely.

## Rollback Plan

If an erroneous activation occurs:

1. **Verify the event** – inspect the Prometheus counter and audit log entry associated with the correlation ID.
2. **Reset the switch** – use the authorised reset workflow (CLI or API) once the incident commander approves.
3. **Reconcile state** – confirm that all execution services report a flat position and that the kill-switch gauge has returned to `0`.
4. **Re-enable trading** – document the root cause, communicate to stakeholders, and monitor the metrics counters for the next 30 minutes to ensure no unexpected re-engagements occur.

Document these steps in your incident runbook so that responders can revert safely under pressure.
