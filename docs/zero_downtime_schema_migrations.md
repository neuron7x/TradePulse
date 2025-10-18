# Zero-Downtime Schema Migrations

Zero-downtime (ZDT) migrations preserve trading uptime while evolving the database. This playbook outlines how to plan, execute, and validate schema changes without interrupting order flow or degrading latency-sensitive workloads.

## Guiding Principles

1. **Two-phase delivery** – Split every breaking change into additive and cleanup phases so old and new application versions can co-exist.
2. **Backward and forward compatibility** – Treat schema, application models, and APIs as contracts. Code must read/write both the pre- and post-migration representations until cleanup completes.
3. **Online execution** – Prefer online DDL primitives (`CONCURRENTLY`, `ONLINE`, `ALGORITHM=INPLACE`) and chunked data backfills to avoid long-running locks.
4. **Observability first** – Instrument migrations with timing, lock metrics, and row counters. Abort if thresholds are exceeded.
5. **Reversibility** – Define explicit fallback and roll-forward paths before touching production.

## Change Design Checklist

- Document the intent, scope, and blast radius in the migration RFC.
- Map every schema change to the application release that consumes it; ensure dual-read/write support where necessary.
- Confirm data-classification tags and row-level security policies remain intact.
- For high-volume tables, size the migration using production statistics (cardinality, bloat, lock contention windows).
- Decide if the change can be performed with online DDL; if not, plan a dual-write shadow table or phased cut-over.
- Validate that index changes use `CREATE INDEX CONCURRENTLY` or engine-specific online options.
- Ensure feature flags or config switches guard behavioral changes until the cleanup phase completes.

## Two-Phase Upgrade Pattern

| Phase | Application State | Schema Actions | Notes |
|-------|-------------------|----------------|-------|
| **Phase 1 – Expand** | Both old and new binaries run | - Add new columns/tables with defaults nullable.<br>- Create new indexes concurrently.<br>- Add triggers or dual-write jobs to backfill data incrementally.<br>- Deploy read-path changes to tolerate both schemas. | Backfill in batches sized by lock time budgets (e.g. 10k rows / 100 ms sleep). Monitor replication lag. |
| **Phase 2 – Contract** | Only new binary remains | - Drop legacy columns/constraints.<br>- Remove compatibility triggers.<br>- Tighten nullability/constraints after verifying metrics for one full trading session. | Confirm rollback plan is no longer needed before contract operations. |

## Execution Workflow

### 1. Pre-migration validation

- Generate a dry-run plan in staging using production-like data snapshots.
- Run `EXPLAIN` on critical queries with the new schema to validate execution plans.
- Ensure application feature flags default to the safe (old) path until backfill completes.
- Schedule the change within approved maintenance windows; capture approval in the change calendar.

### 2. Launch guardrails

- Use the `migration_control` service to acquire a global lock preventing concurrent schema modifications.
- Enable observability dashboards: lock waits, replication lag, queue depths, trading throughput, and error budgets.
- Configure automation thresholds (max lock wait, max batch duration) in the migration job configuration.

### 3. Online migration steps

1. **Additive DDL** – Apply structural changes using online primitives.
2. **Backfill** – Run the data backfill job with adaptive throttling:
   - Limit batch size based on `pg_stat_activity` or engine metrics.
   - Pause automatically if latency SLOs breach or replication lag exceeds the configured ceiling.
   - Emit progress events to the `ops.zdt_migration` topic for audit.
3. **Verification** – Execute the validation suite (see below). If any check fails, initiate the fallback procedure.
4. **Cutover** – Flip feature flags and roll out the new binary via canary then rolling deployment.

### 4. Validation Suite

- **Schema diff** – `SELECT * FROM ops.verify_schema('migration_id');`
- **Constraint & index check** – Ensure all expected indexes exist and are valid (`pg_index.indisvalid = true`).
- **Query replay** – Run captured production queries against the new schema, diffing result sets.
- **Dual-write parity** – Compare shadow table counts and checksums if dual-write is used.
- **Latency monitoring** – Confirm P99 order insert latency remains within SLO.

## Duration Control

- Define a max wall-clock budget per migration step (e.g. additive DDL < 5 min, backfill < 60 min).
- Use chunked backfills with exponential backoff on contention (`max(baseline_sleep, observed_lock_wait * 2)`).
- Abort automatically if exceeding the budget; do not continue without a new change record approval.

## Fallback Procedures

1. **Immediate abort** – Stop the migration job and drop any newly created triggers or temporary tables.
2. **Rollback code** – Redeploy the previously known-good application image from the release registry.
3. **Schema reversion** – Run the pre-generated downgrade script (generated via Alembic `downgrade --sql`).
4. **Data reconciliation** – Use audit logs from `ops.zdt_migration_events` to reconcile partial writes.
5. **Post-mortem** – File an incident report and document remediation before attempting re-run.

## Testing on Production-like Data

- Maintain an anonymised mirror of production in the `shadow` environment; refresh nightly.
- Execute migrations end-to-end against the mirror, including automated validation and performance tests.
- Record timing, lock metrics, and any query plan changes; store results alongside the migration RFC.
- Only promote the migration if mirror results stay within 90% of production latency budgets.

## Monitoring Impact

- Dashboard widgets:
  - `db_lock_wait_seconds` and `db_blocked_sessions`.
  - `trade_throughput_per_minute` with alert if deviation >5% from baseline.
  - Replication lag (`pg_stat_replication.replay_lag`) and CDC stream backlog.
  - Error rate per service (`errors{service=~"trade|risk"}`).
- Create temporary alerts scoped to the migration window; auto-expire after completion.
- Ensure SRE on-call acknowledges the change ticket and monitors dashboards during the window.

## Post-migration Cleanup

- Remove compatibility code paths and feature flags once contract phase completes.
- Run `ANALYZE` on affected tables to refresh statistics.
- Update the schema registry and publish the migration summary in `docs/changelog/database.md`.
- Archive monitoring dashboards and logs for compliance.

## Automation & Tooling

- Extend the migration CI pipeline to require:
  - Static analysis detecting missing two-phase plans or non-online DDL in Alembic migrations.
  - Unit tests covering serialization/deserialization for new models in both schemas.
  - Integration tests replaying historical orders against migrated schemas.
- Store migration manifests in `migrations/manifests/<migration_id>.yaml` describing steps, thresholds, and rollback scripts.

## References

- [Alembic Online DDL Patterns](https://alembic.sqlalchemy.org/en/latest/cookbook.html#online-ddl-example)
- [PostgreSQL Concurrent Index Builds](https://www.postgresql.org/docs/current/sql-createindex.html)
- Internal Runbooks: `docs/incident_playbooks.md`, `docs/governance.md`.

