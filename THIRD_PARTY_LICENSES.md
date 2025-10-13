# Third-Party Licenses

TradePulse depends on the following notable third-party libraries. Each entry
includes the upstream project, license, and a short description of its use in
this repository. This list supplements the SBOM generated in CI.

| Project | License | Usage |
| ------- | ------- | ----- |
| NumPy | BSD-3-Clause | Core numerical routines for analytics, vectorised backtests, and simulations. |
| SciPy | BSD-3-Clause | Optimisation, signal processing, and statistical utilities supporting research workflows. |
| pandas | BSD-3-Clause | DataFrame transformations, ingestion pipelines, and reporting utilities. |
| networkx | BSD-3-Clause | Graph-based market topology analysis and structural metrics. |
| pydantic | MIT | Strongly typed configuration models for CLI commands and strategy definitions. |
| pandera | MIT | Data validation schemas for feature engineering and materialisation. |
| PyYAML | MIT | YAML configuration parsing across CLI templates and automation. |
| hydra-core | BSD-3-Clause | Hierarchical configuration composition for analytics pipelines. |
| prometheus-client | Apache-2.0 | Metrics instrumentation for golden signals and job-specific telemetry. |
| opentelemetry-sdk | Apache-2.0 | Distributed tracing for ingest, backtest, and execution services. |
| click | BSD-3-Clause | Command-line interface for orchestrating ingestion, backtest, train, and publish workflows. |

For a comprehensive inventory, consult the CycloneDX SBOM artifacts emitted by
`.github/workflows/sbom.yml`. Please report licensing concerns to
`security@tradepulse.ai`.
