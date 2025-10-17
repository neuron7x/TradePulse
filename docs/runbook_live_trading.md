# Live Trading Runbook

This runbook documents the operational procedures for running the live execution
stack backed by `execution.live_loop.LiveExecutionLoop`. It is intended for
operators who are responsible for starting, monitoring, and shutting down live
trading sessions.

## Pre-Flight Checklist

1. **Infrastructure readiness** – confirm that market data feeds, network
   routes, and authentication secrets are available for every configured
   exchange connector.
2. **Risk controls** – verify that the `RiskManager` limits reflect the latest
   risk committee directives and that the kill-switch has been reset after the
   previous trading session (use `GET /admin/kill-switch` followed by
   `DELETE /admin/kill-switch` if required).
3. **State directory** – ensure the state directory configured in
   `LiveLoopConfig.state_dir` is writeable and backed up. Historical OMS state
   files are required for warm restarts and forensic analysis.
4. **Metrics and logging** – confirm that the Prometheus collector and log
   aggregation pipelines are running to capture structured events emitted by
   the live loop.

## Cold Start Procedure

Cold starts are used when deploying a new strategy instance or when the prior
state is intentionally discarded.

1. **Initialise the loop**
   ```python
   from execution.live_loop import LiveExecutionLoop, LiveLoopConfig
   from execution.risk import RiskLimits, RiskManager

   config = LiveLoopConfig(state_dir="/var/lib/tradepulse/live")
   risk = RiskManager(RiskLimits(...))
   loop = LiveExecutionLoop({"binance": binance_connector}, risk, config=config)
   loop.start(cold_start=True)
   ```
2. **Hydrate OMS from disk** – the cold start path reloads persisted OMS state
   and clears transient in-memory queues. Any outstanding open orders will be
   re-registered for fill tracking.
3. **Submit orders** – use `loop.submit_order(venue, order, correlation_id)` to
   enqueue new orders. The background submission thread handles placement and
   retries.
4. **Monitor metrics** – dashboards should display the
   `live_loop.*` structured logs, order placement metrics, and per-venue
   heartbeat gauges.

## Warm Start / Restart Procedure

Warm starts resume trading after a controlled shutdown or short outage.

1. **Instantiate with persisted state** – reuse the existing state directory
   and connectors when constructing `LiveExecutionLoop`.
2. **Start in warm mode** – call `loop.start(cold_start=False)`. The live loop
   will:
   - Load persisted OMS state including queued and outstanding orders.
   - Fetch `open_orders()` from each connector to reconcile venue state.
   - Re-enqueue orders that were persisted but missing on the venue.
   - Adopt orphaned venue orders into the OMS to maintain risk accounting.
3. **Validate reconciliation** – check logs for `live_loop.requeue_order` and
   `live_loop.adopt_order` events to confirm that discrepancies were addressed.
4. **Resume trading** – orders can be submitted immediately after the warm
   start completes.

## Monitoring and Observability

- **Structured logs** – the loop emits JSON-friendly logs such as
  `live_loop.order_processed`, `live_loop.register_fill`, and
  `live_loop.heartbeat_retry`. Forward these to the incident dashboard.
- **Metrics** – Prometheus counters and gauges are updated via
  `core.utils.metrics` for order placements, acknowledgements, fills, and
  positions. Ensure dashboards alert on stalled submissions or missing heartbeats.
- **Lifecycle hooks** – subscribe to `on_kill_switch`, `on_reconnect`, and
  `on_position_snapshot` to integrate with alerting or downstream systems.

### API Health Probe Interpretation

- Use `GET /health` before enabling traffic to the inference API or live loop.
  A `200` response with `"status": "ready"` indicates that the risk manager,
  cache, rate limiters, and declared dependencies are within SLO. Any `503`
  response requires intervention before proceeding.
- The `risk_manager` component reports the kill-switch state. When
  `status="failed"` or `healthy=false`, the kill-switch is engaged and the
  detailed reason is surfaced in the `detail` field. Reset the kill-switch via
  the admin API before resuming trading.
- Dependency probes are emitted as `dependency:<name>` components. Failures are
  reported with `status="failed"` and a descriptive message (for example,
  `connection refused`). Investigate upstream services (Kafka, Postgres, market
  data feeds) before retrying.
- `client_rate_limiter` and `admin_rate_limiter` components expose utilisation
  metrics and saturated keys. Repeated saturation should trigger incident
  handling to avoid throttling critical traffic.
- The `inference_cache` component reports occupancy of the TTL cache. A
  `degraded` status indicates the cache is full and requests will skip the fast
  path until entries expire; purge or expand the cache capacity if this state
  persists.

## Kafka Operations

Kafka is now a tier-one dependency for ingesting market data. Coordinate changes
with the data engineering on-call before performing the procedures below.

### Broker Scaling

1. **Review Terraform state** – confirm the desired broker count in
   `infra/terraform/eks/environments/<env>.tfvars` under `msk_config.number_of_broker_nodes`.
2. **Apply infrastructure changes** – run `terraform -chdir=infra/terraform/eks apply -var-file=environments/<env>.tfvars`
   to let the MSK module resize the cluster. Terraform will orchestrate rolling
   replacements that preserve client connectivity.
3. **Update client configuration** – capture the refreshed
   `kafka_bootstrap_brokers_tls` output and update the `KAFKA_BOOTSTRAP_SERVERS`
   environment variable or `configs/live/default.toml`. Redeploy TradePulse if
   the endpoint list changed.
4. **Validate health** – monitor the MSK console for `BrokerNotAvailable` or
   storage pressure alarms and verify that ingestion metrics (`kafka.consumer_lag`
   and `kafka.records_per_second`) stabilise within five minutes.

### Partition Reassignment

1. **Plan reassignment** – export the current partition layout using
   `kafka-topics.sh --bootstrap-server $BOOTSTRAP --describe --topic tradepulse.market-data`.
2. **Generate proposal** – create a JSON reassignment plan sized for the new
   broker set. Use `kafka-reassign-partitions.sh --bootstrap-server $BOOTSTRAP \
   --topics-to-move-json-file plan.json --generate`.
3. **Apply** – execute the reassignment with the generated JSON and monitor
   progress (`--verify`) until completion. Ensure each partition has at least two
   in-sync replicas; if replication falls behind, pause producers until ISR
   recovers.
4. **Terraform sync** – document the updated replication targets in
   `msk_config.configuration_properties` (for example `default.replication.factor`)
   so that future applies preserve the new layout.

### Lag Monitoring and Remediation

1. **Dashboards** – watch the ingestion lag panels sourced from
   `KafkaIngestionService` hot cache metrics. Spikes above 60 seconds should page
   the on-call.
2. **Correlate with MSK metrics** – inspect `aws.msk.brokerTopicMetrics.*` in
   CloudWatch (enabled by the Terraform module). Focus on `MessagesInPerSec` and
   `BytesInPerSec` per partition.
3. **Execute lag drain** – apply a temporary override to
   `KafkaIngestionConfig.max_batch_size` and `linger_ms` via the live config
   (`configs/live/default.toml`) and redeploy ingestion to accelerate catch-up.
   Scale the values back once lag returns to baseline to minimise memory
   pressure.
4. **Escalate** – if lag persists beyond 10 minutes, engage the infrastructure
   team to review broker CPU and storage. Scaling guidance is in the section
   above.

## Failure Handling

- **Connector disconnects** – heartbeat failures trigger exponential backoff
  retries and emit `on_reconnect` events. Investigate repeated retries and be
  prepared to fail over if the venue remains unreachable.
- **Order discrepancies** – warm start reconciliation produces warnings when
  orders are re-queued or adopted. Operators should confirm that downstream
  systems (P&L, hedging) reflect the corrected state.
- **Kill-switch activation** – when the risk kill-switch triggers, the live loop
  stops all background tasks, emits `on_kill_switch`, and requires manual
  intervention before restarting. Investigate the root cause, confirm the
  current status with `GET /admin/kill-switch`, and only resume once
  `DELETE /admin/kill-switch` records a successful reset event.

## Shutdown Procedure

1. Call `loop.shutdown()` to stop background workers and disconnect connectors.
2. Confirm that no orders remain queued and that OMS state files were persisted.
3. Archive logs and metrics for the session as part of the post-trade review.

## Contacts and Escalation

- **Execution engineering on-call** – primary contact for connector incidents.
- **Risk management** – escalation path for kill-switch or limit breaches.
- **Operations** – responsible for scheduling and communication during planned
  maintenance windows.

Keep this runbook up to date alongside changes to `execution/live_loop.py` and
related operational tooling.
