# Live Trading Runbook

This runbook documents the operator procedures for starting, monitoring, and
safely shutting down the live execution loop. It assumes the
`LiveExecutionLoop` orchestrator introduced under `execution/live_loop.py`.

## Cold Start Procedure

A cold start should be used when bootstrapping a new trading session or after
performing maintenance that invalidates local execution state.

1. **Review change windows** – confirm the trading window is clear of market
   events that would violate the desk's risk appetite.
2. **Prepare credentials** – export venue API keys and network allow-list
   entries in the runtime environment.
3. **Bootstrap services** – ensure data ingestion, strategy runners, and risk
   services are healthy before enabling execution.
4. **Invoke the loop** – instantiate `LiveExecutionLoop` and call
   `start(cold_start=True)`. The loop performs the following steps:
   - Connects to every configured execution connector with exponential
     backoff.
   - Purges persisted OMS state and resets the risk kill-switch.
   - Spins up background workers for order submission, fill polling, and
     heartbeat monitoring.
5. **Verify readiness** – tail structured logs for `event="cold_start"` and
   confirm metrics such as `tradepulse_open_positions` report zeroed state.
6. **Enable strategies** – once the loop is live, release strategy signals into
   the queue.

## Warm Start / Restart Procedure

Warm starts are used during controlled restarts where existing orders must be
reconciled.

1. **Quiesce strategies** – pause new order generation to avoid duplicate
   submissions.
2. **Restart the loop** – call `start(cold_start=False)` on a new
   `LiveExecutionLoop` instance.
3. **State hydration** – the loop reloads OMS state from disk, fetches
   `open_orders()` from each connector, adopts venue orders that are missing
   locally, and re-enqueues orphaned OMS orders for resubmission.
4. **Position snapshot** – monitor emitted position callbacks/logs to confirm
   exposure matches desk records.
5. **Resume strategies** – after reconciliation logs
   `event="warm_start"`, re-enable signal generation.

## Monitoring & Alerting

- **Heartbeats** – heartbeat workers call `get_positions()` on each connector
  and emit structured logs with `event="heartbeat"`. Alert if no heartbeat is
  observed within two heartbeat intervals.
- **Reconnection events** – reconnection callbacks signal when the loop had to
  recover connectivity. Persistent reconnections should page engineering.
- **Kill-switch triggers** – any critical failure or risk violation triggers
  the kill-switch and emits `event="kill_switch"`. Operators must investigate
  the root cause and reset the kill-switch only after remediation.
- **Metrics** – scrape Prometheus metrics (if enabled) for order latency,
  queue depth, and open positions to ensure they remain within operational
  thresholds.

## Shutdown Procedure

1. **Cease order intake** – instruct strategies to stop enqueuing orders.
2. **Drain queue** – allow the order submission worker to finish outstanding
   placements or cancel as necessary.
3. **Call `shutdown()`** – the loop stops workers, disconnects connectors, and
   logs `event="loop_shutdown"`.
4. **Post-mortem** – archive logs and metrics for audit, especially if the
   kill-switch was triggered.

## Incident Response

- **Connector outage** – the loop automatically retries connections with
  exponential backoff. If the outage persists beyond desk tolerance, trigger
  the kill-switch and hedge positions manually.
- **Data inconsistencies** – if position snapshots diverge from ledger
  records, halt trading via the kill-switch, reconcile with the venue, and
  warm-start once the discrepancy is resolved.
- **Unexpected fills** – the fill polling worker records partial fills. If
  fills cannot be matched to OMS orders, escalate to engineering and risk for
  investigation before resuming trading.

Maintaining this runbook ensures that live trading operations remain safe,
repeatable, and auditable.
