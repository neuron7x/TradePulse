# Rate limiting and host protection

TradePulse now exposes configurable knobs for the operations team to tune per-client
quotas and ingress filtering. This document summarises the key environment variables
and the recommended workflow when rolling out changes.

## Per-client rate policies

Rate limits are managed by `ApiRateLimitSettings` which reads configuration from the
`TRADEPULSE_RATE_*` environment variables.

| Setting | Environment variable | Description |
| --- | --- | --- |
| `default_policy` | `TRADEPULSE_RATE_DEFAULT_POLICY` | JSON object defining `max_requests` and `window_seconds` for authenticated subjects without a dedicated override. |
| `unauthenticated_policy` | `TRADEPULSE_RATE_UNAUTHENTICATED_POLICY` | Optional JSON object that constrains unauthenticated/IP keyed requests. When omitted the default policy applies. |
| `client_policies` | `TRADEPULSE_RATE_CLIENT_POLICIES` | JSON object mapping authenticated subjects (API keys or mTLS subjects) to individual policies. |
| `redis_url` | `TRADEPULSE_RATE_REDIS_URL` | Redis URI used to coordinate limits across multiple application instances. |
| `redis_key_prefix` | `TRADEPULSE_RATE_REDIS_KEY_PREFIX` | Prefix applied to Redis keys when coordinating quotas. |

### Example: dedicate a burstier policy to a specific subject

```bash
export TRADEPULSE_RATE_DEFAULT_POLICY='{"max_requests": 120, "window_seconds": 60}'
export TRADEPULSE_RATE_CLIENT_POLICIES='{"vip-client": {"max_requests": 240, "window_seconds": 60}}'
export TRADEPULSE_RATE_REDIS_URL='redis://cache.internal:6379/0'
```

After updating the environment, redeploy the service. The FastAPI application will
initialise a Redis-backed sliding-window limiter automatically when `redis_url` is
provided.

### Operational checklist

1. Update the appropriate `TRADEPULSE_RATE_*` variables for your environment.
2. Ensure Redis connectivity (TLS, authentication) is configured before rolling
   out a clustered deployment.
3. Restart the application pods and monitor the `429` metrics for unexpected spikes.
4. Document subject-specific overrides in your runbooks to help support teams.

## Host allow-list and payload inspection

`ApiSecuritySettings` now includes defensive defaults that can be overridden via
environment variables with the `TRADEPULSE_` prefix.

| Setting | Environment variable | Description |
| --- | --- | --- |
| `trusted_hosts` | `TRADEPULSE_TRUSTED_HOSTS` | Comma-separated list or JSON array of host headers accepted by the API gateway. |
| `max_request_bytes` | `TRADEPULSE_MAX_REQUEST_BYTES` | Maximum payload size (in bytes) accepted before rejecting with HTTP 413. |
| `suspicious_json_keys` | `TRADEPULSE_SUSPICIOUS_JSON_KEYS` | JSON array of JSON keys that should trigger a rejection. |
| `suspicious_json_substrings` | `TRADEPULSE_SUSPICIOUS_JSON_SUBSTRINGS` | JSON array of substrings that, when present in JSON values, cause the WAF to reject the request. |

### Example: tighten ingress for production

```bash
export TRADEPULSE_TRUSTED_HOSTS='["api.tradepulse.com", "api.tradepulse.internal"]'
export TRADEPULSE_MAX_REQUEST_BYTES=524288
export TRADEPULSE_SUSPICIOUS_JSON_KEYS='["$where", "__proto__"]'
export TRADEPULSE_SUSPICIOUS_JSON_SUBSTRINGS='["<script", "javascript:"]'
```

Apply the new configuration and verify that your ingress controller forwards the
correct host headers. Requests originating from unlisted hosts or violating the
payload rules will be rejected before reaching any route handlers.

### Operational checklist

1. Confirm that health checks and reverse proxies use an allowed host header.
2. Roll out stricter payload limits during off-peak hours and monitor error logs
   for false positives.
3. Review suspicious payload patterns regularly and update the JSON guard lists
   as new attack signatures emerge.

## Testing and validation

* Use `pytest tests/api/test_service.py::test_client_rate_limit_is_enforced` to
  validate rate limit policies locally.
* Use `pytest tests/api/test_service.py::test_trusted_host_middleware_blocks_unlisted_hosts`
  and `pytest tests/api/test_service.py::test_payload_guard_rejects_large_and_suspicious_bodies`
  to validate WAF behaviour before promoting changes.
