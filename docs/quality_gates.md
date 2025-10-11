# Quality Gates and Automated Governance

Quality gates prevent regressions from entering `main` by enforcing measurable
criteria across testing, performance, and reliability domains.

## Pull Request Blocking Rules

1. **Code Coverage** – PRs fail if overall coverage drops by more than 0.5% or
   critical packages fall below 85%. Coverage deltas are computed against the
   baseline stored in `reports/coverage-baseline.json`.
2. **Performance Budget** – Frontend builds must satisfy the budgets defined in
   `docs/performance.md`. Lighthouse CI asserts route-level LCP/TTFB, while the
   bundle-size bot compares asset weights with the previous successful build.
3. **Latency SLOs** – Synthetic smoke tests replay key execution flows. If the
   SLI dashboards predict burn-rate exhaustion under 72 hours, the PR merge is
   blocked until mitigations land.
4. **Heavy-math validation** – Dedicated jobs defined in
   `configs/quality/heavy_math_jobs.yaml` execute Kuramoto, Ricci, and Hurst
   stress workloads with enforced CPU/memory quotas. The jobs run under the
   `heavy_math` pytest marker and must pass before a PR can merge.
5. **Cross-architecture parity** – Indicator portability tests labelled with the
   `arm` marker compare CPU, GPU, and float32 (ARM-simulated) execution paths.
   CI requires parity within ±0.005 absolute tolerance; deviations block the
   pipeline and page the platform quality channel.

## Nightly Benchmarks and Auto-Triage

- Nightly benchmark jobs compare runtime metrics (execution latency, backtest
  throughput, ingestion lag) to the last green build.
- When degradation exceeds configured thresholds (default 5%), the pipeline:
  - Adds an `auto-regression` label to the offending PR if identified.
  - Creates a Jira task in the `Reliability` project assigned to the owning team.
  - Posts a summary to `#eng-quality` including CLI artifact hashes for the run.

## Workflow for Exceptions

1. Engineer raises an exception request with justification, impact analysis, and
   expiry date.
2. Quality lead reviews and, if approved, records the exception in
   `configs/quality/exceptions.yaml` with a maximum lifetime of 14 days.
3. CI pipelines surface active exceptions in PR summaries to ensure temporary
   allowances receive scrutiny.

## Integration with CLI Outputs

- CLI commands emit SHA256 hashes and JSONL streams; quality bots record these to
  prove that benchmark inputs/outputs were unchanged.
- When running `tradepulse-cli backtest --output jsonl`, the pipeline pipes the
  results through `jq` scripts that calculate regression deltas and post them to
  the run summary.

## Reporting

- Weekly reports aggregate gate outcomes, open exceptions, and regression trends
  into the Quality dashboard.
- Error budget consumption from `docs/reliability.md` is overlaid with gate
  statuses so leadership can prioritise fixes versus feature work.

Quality gates shift enforcement left: developers receive immediate feedback in
PRs, nightly automation provides early detection, and governance artifacts keep
stakeholders aligned.
