# TradePulse Roadmap

The roadmap translates the long-term improvement plan into a time-phased delivery schedule. It highlights the initiatives
required to keep TradePulse production-grade while enabling sustainable growth. Progress should be reviewed monthly and the
roadmap updated after every major milestone.

## 2024 Outlook

| Quarter | Focus Areas | Milestones |
| --- | --- | --- |
| **Q2 2024** | Architecture, Developer Experience, Typing | • Publish C4 system diagram and module dependency graph.<br>• Ship VS Code dev container with automated onboarding.<br>• Enable strict mypy for `core` and `libs/utils` packages.<br>• Introduce plugin loader prototype for strategies. |
| **Q3 2024** | Testing, Observability, Backtesting | • Raise unit test coverage for `core` and `libs/utils` to ≥90%.<br>• Add dockerised end-to-end scenario harness covering ingestion → execution flow.<br>• Version dashboards and alert rules in `observability/` with CI linting.<br>• Extend backtester with commission and slippage models parameterised via configs. |
| **Q4 2024** | Security, Performance, Culture | • Publish SBOM artifacts alongside every release.<br>• Integrate profiling scripts and publish baseline latency/throughput reports.<br>• Launch Dependabot and Trivy gates in CI.<br>• Document chaos testing playbooks and schedule quarterly GameDays. |

## 2025 North Star Themes

- **Extensible Architecture**: Complete the pluggable strategy framework with entry-point discovery, version negotiation, and
  compatibility validation. Pair with API versioning and OpenAPI specs for external partners.
- **Holistic Testing**: Maintain a regression test matrix that ties features to unit, integration, property-based, and
  performance suites. Automate nightly E2E scenarios with mocked exchanges and stress backtest scenarios (flash crashes,
  trading halts).
- **Production Observability**: Define explicit SLOs, implement burn-rate alerts, and expand runbooks with escalation paths and
  troubleshooting dashboards.
- **Secure Supply Chain**: Embed SBOM generation, SAST/DAST gates, and dependency hygiene automation in the release pipeline to
  guarantee <7 day turnaround on critical CVEs.
- **Performance & Scalability**: Establish continuous profiling, benchmarking, and adaptive worker scaling so TradePulse meets
  latency targets under variable load.
- **Engineering Excellence**: Keep documentation current (CHANGELOG, roadmap, architectural diagrams) and maintain contributor
  guidelines that facilitate external collaboration.

## How to Use This Roadmap

1. **Plan sprints**: Reference the current quarter milestones when creating sprint goals and cross-team commitments.
2. **Track progress**: Update milestone status (Not Started → In Progress → Done) in pull requests and release notes.
3. **Align stakeholders**: Share the roadmap during product reviews and incident postmortems to keep expectations realistic.
4. **Revisit quarterly**: Review the improvement plan and adjust target quarters or milestones based on capacity and impact.
5. **Document decisions**: Capture significant scope changes and rationale in `CHANGELOG.md` and link back to the roadmap.

Maintaining this roadmap ensures the "development map" for TradePulse remains actionable, transparent, and aligned with the
platform's strategic objectives.
