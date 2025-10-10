# Dataset Catalog

This catalog summarises the curated datasets that live in the `observability/` metrics registry and the `reports/` knowledge base. Each entry documents the dataset's purpose, provenance, temporal coverage, and current quality caveats to streamline governance and discovery.

## Observability Metrics Registry

| Dataset | Location | Description | Source | Temporal Coverage | Quality Flags |
| --- | --- | --- | --- | --- | --- |
| Feature Transformation Metrics | `observability/metrics.json` (`tradepulse_feature_*` series) | Defines histograms, counters, and gauges tracking feature engineering runtimes, execution counts, and latest values per feature. | Internal instrumentation emitted by the feature service. | Applies to live telemetry streamed via Prometheus; definitions are time-range agnostic but target real-time and rolling retention windows. | Schema validated in repo; ensure exporters remain aligned when metrics are renamed. |
| Backtesting Metrics | `observability/metrics.json` (`tradepulse_backtest_*` series) | Captures wall-clock duration, run counts, P&L, drawdown, and trade volumes for strategy backtests. | Backtest engine instrumentation. | Real-time for ongoing runs; historical range governed by observability retention policies. | Counters reset on service restart; annotate dashboards when resets occur. |
| Data Ingestion Metrics | `observability/metrics.json` (`tradepulse_data_ingestion_*`, `tradepulse_ticks_processed_total`) | Tracks ingestion durations, task counts, and processed tick volumes by source/symbol. | Data ingestion workers publishing Prometheus metrics. | Continuous telemetry; align retention with data lake SLAs. | Requires source label normalisation to avoid cardinality explosions. |
| Execution Metrics | `observability/metrics.json` (`tradepulse_order_*`, `tradepulse_open_positions`) | Measures order placement latency, order totals, and open position counts across exchanges and symbols. | Execution service instrumentation. | Near real-time order flow; historical trends depend on Prometheus retention. | Latency histograms rely on consistent bucket configuration across environments. |
| Strategy Runtime Metrics | `observability/metrics.json` (`tradepulse_strategy_*`) | Tracks live strategy scores and memory footprint. | Strategy orchestration service. | Real-time monitoring with rolling aggregation windows. | Confirm score calculation parity across staging and production clusters. |
| Optimisation Metrics | `observability/metrics.json` (`tradepulse_optimization_*`) | Logs optimisation duration histograms and iteration counters for strategy tuning workloads. | Optimisation engine instrumentation. | Streaming telemetry per optimisation job; historical coverage subject to retention policies. | Counters reset between jobs; annotate monitoring to distinguish cumulative vs per-run views. |

## Operational Reports Repository

| Dataset | Location | Description | Source | Temporal Coverage | Quality Flags |
| --- | --- | --- | --- | --- | --- |
| CI/CD Health Review | `reports/ci_cd_health_review.md` | Narrative summary of current CI test status, warning analysis, and remediation recommendations. | Authored from latest pipeline inspection. | Snapshot of most recent CI evaluation; update with each health review cycle. | Manual report—requires validation against latest pipeline logs before distribution. |
| Coverage Analysis Report | `reports/coverage_analysis_report.md` | Highlights code coverage metrics and key improvement areas. | Derived from coverage tooling outputs. | Represents the coverage state at last analysis run. | Replace figures if coverage thresholds or tooling change. |
| Release Readiness Report | `reports/release_readiness.md` | Consolidates release risks, outstanding tasks, and documentation gaps. | Release management review. | Point-in-time assessment ahead of next release gate. | Requires refresh per release candidate; flag stale recommendations. |
| Technical Debt Assessment | `reports/technical_debt_assessment.md` | Qualitative review of principal technical debt items and mitigation plans. | Architecture working group findings. | Current sprint planning horizon. | Cross-check items against backlog to ensure alignment. |
| Complexity Snapshot | `reports/complexity.json` | Machine-generated complexity metrics for tracked modules. | Static analysis pipeline export. | Based on last analysis execution timestamp embedded in pipeline logs. | Confirm schema version before ingest; mark runs missing module coverage. |

## Usage Notes

- **Metadata stewardship**: Owners should update this catalog whenever new datasets or metrics are introduced or existing ones are deprecated.
- **Temporal coverage**: Where exact ranges are unknown, rely on the upstream pipeline or monitoring retention configuration and document changes here.
- **Quality governance**: Quality flags highlight known caveats; convert them into tracked remediation tasks when persistent.
