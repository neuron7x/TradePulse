# Quality Assurance Playbook

This playbook defines the iterative quality loop required before tagging a TradePulse release. It combines dependency verificat
ion, integration coverage, and optimisation guardrails so that every build can graduate from local testing to production deploym
ent with confidence.

## 1. Objectives
- **Stability.** Ship artefacts that install, configure, and execute without manual patching.
- **Integrity.** Maintain alignment between documentation claims, runtime capabilities, and automated checks.
- **Performance awareness.** Track computational cost of indicators, backtests, and data pipelines to prevent regressions.

## 2. Environment Baseline
1. Create a clean virtual environment (Python 3.11+).
2. Install runtime dependencies and testing extras:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Cache heavyweight dependencies (NumPy, pandas, PyYAML, hypothesis) in CI to minimise cold-start time.
4. Export the environment with `pip freeze > artifacts/requirements.lock` for reproducibility.

## 3. Integration Loop
Each development iteration must complete the following cycle:

| Step | Command | Purpose |
| ---- | ------- | ------- |
| 1 | `pytest -q` | Validate unit, integration, property, and fuzz suites with full dependencies. |
| 2 | `pytest --cov=core,backtest,execution --cov-report=xml --cov-fail-under=98` | Enforce the documented coverage threshold. |
| 3 | `python -m interfaces.cli backtest configs/backtests/sample.yaml` | Smoke-test YAML-driven CLI workflows. |
| 4 | `make lint` | Run static analysis (ruff, mypy, gofmt) and ensure style consistency. |
| 5 | `make docs` | Confirm MkDocs builds and cross-links documentation updates. |

## 4. Optimisation & Regression Checks
- Track execution time of key strategies using `pytest --durations=20` to flag slow tests.
- Add benchmark fixtures for critical indicators (e.g., multiscale Kuramoto, temporal Ricci) and monitor changes via CI trends.
- Profile memory usage with `pytest --maxfail=1 --pdb` when Hypothesis generates large datasets.
- Document performance findings in `reports/technical_debt_assessment.md` to maintain institutional knowledge.

## 5. Release Gate Checklist
Before opening a release candidate PR:
- ✅ Dependencies resolved and locked; clean checkout reproduces the environment.
- ✅ Test matrix passing locally and in CI with coverage ≥ 98 %.
- ✅ Release notes document any intentional deviations or experimental modules.
- ✅ Monitoring and alerting dashboards updated to match metric names in code.
- ✅ Rollback procedure rehearsed and documented in deployment runbooks.

## 6. Feedback Integration
- Capture post-mortem notes from failed iterations and append mitigation tasks to the backlog.
- Update this playbook whenever new tooling, metrics, or performance targets are introduced.
- Review the playbook quarterly to ensure alignment with evolving architecture and market requirements.

Following this playbook enforces disciplined, iterative quality improvements and keeps TradePulse on a predictable path to a prod
uction-ready release.
