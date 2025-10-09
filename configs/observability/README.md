# TradePulse Observability Assets

This directory contains reusable observability collateral that aligns with the
unified logging/metrics/tracing stack.

- `grafana/`: Importable dashboards designed for the default Prometheus metric
  names (`tradepulse_*`).
- Additional exporters, runbooks, and alert rules can be added next to this
  README to keep observability artefacts in one place.

## Dashboards

The Grafana dashboards assume the Prometheus scrape job is named
`tradepulse` and that metrics are exposed through
`bootstrap_observability(...)` defaults.

Import them via the Grafana UI (`+` ➜ *Import*) or by mounting the JSON files
inside your provisioning pipeline.
