# Runbook: Live Trading Operations

This runbook provides step-by-step guidance for launching, monitoring, and safely
halting live trading sessions in TradePulse. It is structured as a checklist to
minimise ambiguity during critical operations.

## 1. Preconditions and Environment Checks

1. **Change approvals** – Confirm the deployment ticket is approved and the
   signed strategy manifest is present in `reports/change_manifest/`.
2. **Health gate** – Verify the latest CI build is green, including heavy-math
   gates and cross-architecture indicator comparisons (see [Testing Guide](../TESTING.md)).
3. **Market readiness** – Inspect the market calendar for the target venues. If a
   holiday or trading halt is active, postpone activation and communicate via
   `#trading-ops`.
4. **Risk envelope** – Ensure guardrails in `configs/risk/limits.yaml` reflect
   the approved exposure (per-instrument notional, leverage caps, kill thresholds).

## 2. Launch Procedure

| Step | Action | Owner | Telemetry |
| ---- | ------ | ----- | --------- |
| 1 | Lock the deployment in the orchestrator to prevent concurrent edits. | Execution Trader | `orchestrator.lock` audit log |
| 2 | Run `tradepulse-cli deploy --env prod --strategy <id>` with the approved
    artifact digest. | Execution Trader | CLI output archived in `reports/live/<date>/deploy.log` |
| 3 | Validate strategy parameters against the governance service. Two-factor
    confirmation is required for the final promote step. | Execution Trader + Risk Officer | `governance.strategy_changes` topic |
| 4 | Trigger smoke validation: replay last 15 minutes of market data in
    dry-run mode to ensure orders remain within tolerance bands. | Quant On Call | `reports/live/<date>/smoke.json` |
| 5 | Flip the `live.enabled` feature flag through the control plane UI. | Execution Trader | Feature flag audit trail |

**Go/No-Go Criteria**

- Telemetry dashboards (`observability/dashboards/live.yaml`) show healthy order
  acknowledgements and no queue backlog.
- Risk service confirms effective limits and kill-switch wiring via `risk/live/status`.
- Governance topic contains a signed change event with matching digest.

## 3. Routine Monitoring

1. **Heartbeat checks** – Streaming ingestion, feature store, execution gateways,
   and order acknowledgements must report within SLA (1 minute for market data,
   2 seconds for execution).
2. **Latency guardrails** – Order round-trip latency must remain below 120 ms.
   Breaches trigger automated throttling and Slack alerts.
3. **Position drift** – Compare live positions against portfolio targets every
   minute. Differences >0.5% notional require manual review.
4. **Risk telemetry** – Ensure kill-switch subscriptions are active and risk
   service heartbeat is green. Missing heartbeats for 3 cycles mandate
   pre-emptive trading halt.

## 4. Planned Stop Procedure

1. Announce planned halt in `#trading-ops` and update the trading calendar.
2. Flip `live.enabled` feature flag to `false`. Capture the resulting audit event.
3. Wait for open orders to settle (monitor until `open_orders` metric is zero).
4. Run `tradepulse-cli settle --strategy <id>` to flush residual state.
5. Archive run artefacts: order logs, PnL snapshots, risk metrics.
6. Record stop confirmation in the governance ledger referencing ticket ID.

## 5. Emergency Scenarios

### 5.1 Kill-Switch Activation

1. Trigger kill-switch via the hardware console or
   `tradepulse-cli kill --strategy <id> --reason <text>`.
2. Confirm the kill event appears in `observability/audit/kill_events.jsonl` with
   dual signatures (initiator + confirmer).
3. Notify `#trading-ops`, `#risk`, and the on-call engineer. Include expected
   recovery timeline.
4. Freeze further deployments until post-mortem sign-off.

### 5.2 Market Data Outage

1. Switch strategies into safe mode (halt new orders, cancel resting ones).
2. Engage the data incident runbook (see [Data Incident Runbook](runbook_data_incident.md)).
3. Maintain manual oversight of risk exposures until feeds stabilise.

### 5.3 Execution Venue Instability

1. Throttle order rate using execution control plane (set `max_orders_per_sec`
   to mitigation value).
2. Contact venue support and log ticket ID in `reports/live/<date>/venue_incidents.md`.
3. If instability persists >5 minutes or threatens risk posture, execute
   kill-switch procedure.

## 6. Communication Matrix

| Trigger | Audience | Channel | Template |
| ------- | -------- | ------- | -------- |
| Launch start | Trading Ops, Risk, Quant On Call | `#trading-ops` | `Runbook: Launching <strategy>` |
| Kill-switch activation | Executive, Compliance, Trading Ops | PagerDuty bridge + `#incidents` | `Kill-switch triggered for <strategy> due to <reason>` |
| Planned halt | Trading Ops, Risk | `#trading-ops` | `Runbook: Stopping <strategy>` |
| Recovery complete | Trading Ops, Risk, Compliance | `#trading-ops` + email | `Strategy <id> restored at <time>, metrics nominal` |

## 7. Post-Run Activities

1. Submit annotated summary to `reports/live/<date>/summary.md`.
2. Update telemetry dashboards with any threshold adjustments.
3. File retrospective issues for automation gaps or manual toil.
4. Review audit logs ensuring all 2FA confirmations and signatures are present.

## 8. Checklist Summary

- [ ] Approvals verified and manifests signed
- [ ] CI, heavy-math, and cross-architecture tests green
- [ ] Launch smoke tests completed and archived
- [ ] Communications executed per matrix
- [ ] Audit artefacts stored and linked to ticket
- [ ] Post-run review completed

Keeping this runbook current is mandatory; review quarterly and after every
live-trading exercise.
