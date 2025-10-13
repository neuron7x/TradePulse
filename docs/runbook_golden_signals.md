# Golden Signal Runbook

This runbook maps TradePulse golden-signal alerts to investigation and
remediation steps. Every responder should keep these actions close to ensure
MTTR stays below roadmap targets.

## TradePulseRequestErrorRate
- **Trigger**: `tradepulse_request_errors_total / tradepulse_http_requests_total > 2%` for 10 minutes.
- **Triage**
  - Inspect the CI HTML publish report for the failing commit to confirm recent
    dependency or configuration changes (`reports/ci-html/publish_report.html`).
  - Check rollout status via the deployment dashboard and halt progressive
    delivery if the increase aligns with a new release.
  - Examine upstream dependency status (broker APIs, feature stores) for active
    incidents.
- **Mitigation**
  - Initiate automated rollback via deployment playbooks when error rate >5%.
  - If external dependency outage, enable circuit breakers to shed load and
    switch ingest jobs to replay mode.
- **Verification**
  - Confirm error rate drops below 1% for 10 consecutive minutes before closing
    the incident.

## TradePulseLatencyP99
- **Trigger**: `histogram_quantile(0.99, tradepulse_request_duration_seconds)`
  exceeds 1.5 seconds for 15 minutes.
- **Triage**
  - Review `observability/alerts.json` for correlated latency alerts (order
    placement, signal-to-fill).
  - Inspect autoscaling metrics; saturation often correlates with insufficient
    capacity or noisy neighbours.
  - Validate downstream services (feature store, model serving) have not
    introduced new throttling rules.
- **Mitigation**
  - Scale out affected workloads; ensure HPA minimum replicas are adequate for
    burst traffic.
  - Prioritise latency-sensitive queues by pausing non-critical research jobs.
  - If caused by external dependencies, apply request hedging or fallback
    caching.
- **Verification**
  - Observe p95 and p99 latency returning to baseline (< 800ms) for 3 windows
    before resuming paused workloads.

## TradePulseSaturationHigh
- **Trigger**: `tradepulse_service_saturation{resource="cpu"} > 0.85` for 10
  minutes.
- **Triage**
  - Check the CI backtest HTML report for unusually large optimisation sweeps or
    heavy research loads.
  - Verify node autoscaling events and inspect container limits for throttling.
  - Review recent chaos experiments; ensure load tests were not left running.
- **Mitigation**
  - Increase replica counts or move compute-heavy workloads to dedicated nodes.
  - Adjust queue concurrency limits to smooth bursty job submissions.
  - Engage the platform team if saturation persists after scaling.
- **Verification**
  - Saturation must remain below 70% for 15 minutes before clearing the alert.

## TradePulseEventLoopBacklog
- **Trigger**: `tradepulse_event_loop_backlog > 50` for 5 minutes.
- **Triage**
  - Inspect async ingestion logs for slow database writes or network partitions.
  - Check chaos-test dashboards for ongoing failure injection.
  - Confirm no long-running synchronous work is blocking the event loop (profile
    using `asyncio.run(debug=True)`).
- **Mitigation**
  - Restart affected pods with elevated logging to capture stuck coroutines.
  - Redirect ingest/materialisation jobs to standby replicas until backlog
    clears.
  - If caused by downstream slowness, apply backpressure by slowing producers.
- **Verification**
  - Ensure backlog falls below 10 tasks and remains stable for 10 minutes.

---

Alert configurations live in `observability/alerts.json`. Update both the
Prometheus rules and this runbook when adding or modifying golden signal
coverage.
