# Test Coverage & Quality Gates Guide

This guide explains how the main CI workflow (`.github/workflows/ci.yml`) enforces coverage and
quality gates, and how to wire the resulting checks into branch protection rules.

## Overview

The CI workflow runs a multi-stage pipeline:

1. **Lint & formatting** via a reusable pre-commit workflow (`lint.yml`).
2. **Static analysis** with `mypy` and `slotscheck`.
3. **Security audits** of Python dependencies (`pip-audit` and `safety`).
4. **Parallelised test matrix** across Python 3.11 and 3.12 using `pytest-xdist`, `pytest-rerunfailures`,
   and coverage reporting.
5. **Flaky test quarantine** that retries quarantined suites.
6. **Coverage gate enforcement** that aggregates line/branch coverage and fails if thresholds are not met.
7. **Codecov uploads** for historical trends and delta checks.

The default thresholds are **92% line coverage** and **85% branch coverage**. Both the `pytest`
invocation and the `coverage-gate` job enforce these limits, so a regression fails fast.

## Configuration

### Adjusting coverage thresholds

Thresholds are configured once in `ci.yml` through the environment variables
`COVERAGE_LINE_THRESHOLD` and `COVERAGE_BRANCH_THRESHOLD`:

```yaml
env:
  COVERAGE_LINE_THRESHOLD: '92'
  COVERAGE_BRANCH_THRESHOLD: '85'
```

When you update these values:

1. Keep the numbers in sync with branch protection expectations and the `coverage-gate` job.
2. Update this README so contributors understand the new policy.

### Customising coverage scope

By default we instrument the core Python packages:

```bash
pytest \
  --cov=analytics --cov=application --cov=backtest --cov=core \
  --cov=domain --cov=execution --cov=interfaces --cov=markets \
  --cov=observability --cov=tools
```

Add or remove packages inside the `pytest` step if your project layout changes.

### Codecov integration

The workflow uploads `coverage.xml` for each Python version via `codecov/codecov-action@v4` with
`fail_ci_if_error: true`. For public repositories, no token is required. For private repositories,
set a `CODECOV_TOKEN` secret under **Settings → Secrets and variables → Actions**.

To enable the optional "no negative delta" policy:

1. Create (or update) a `codecov.yml` configuration in the repository root with:
   ```yaml
   coverage:
     status:
       project:
         default:
           target: auto
           threshold: 0%
           informational: false
       patch:
         default:
           target: 0%
           threshold: 0%
           informational: false
   ```
2. In the Codecov UI, enable the **"Require Changes"** toggle for both project and patch checks.
3. Add the generated Codecov status checks (e.g. `codecov/project` and `codecov/patch`) to your
   branch protection rules (see below).

## Branch protection setup

To require CI, coverage, and Codecov checks before merging:

1. Navigate to **Settings → Branches** and edit or create the rule for `main` (and `develop` if used).
2. Enable **Require a pull request before merging** and **Require status checks to pass before merging**.
3. Add the following checks:
   - `CI / Style & lint (pre-commit)`
   - `CI / Static type checking`
   - `CI / Dependency security scan`
   - `CI / Test matrix (Python 3.11)`
   - `CI / Test matrix (Python 3.12)`
   - `CI / Coverage gate enforcement`
   - `CI / Flaky test quarantine` (optional but recommended so quarantined flakes stay green)
   - `CodeQL (Python)`
   - `codecov/project` and `codecov/patch` (after enabling in Codecov)

You may also require the `Security Scan` workflow (Bandit/Safety/Trivy) for elevated assurance.

## Viewing coverage artefacts

### GitHub Actions summary

Each `Test matrix` job and the `Coverage gate enforcement` job append tables to the run summary. The
reports include line and branch coverage percentages alongside their thresholds.

### Downloadable artefacts

For every Python version the workflow uploads:

- `junit-py3x.xml` — machine-readable test results.
- `coverage-py3x` — HTML coverage reports plus `coverage.xml`.
- `coverage-metrics-py3x.json` — the metrics consumed by the gate job (useful for debugging).

Artefacts are retained according to your repository settings (default: 90 days).

### Codecov dashboards

Visit `https://app.codecov.io/gh/<owner>/<repo>` for interactive visualisations, file-by-file
coverage, and PR deltas. Configure Codecov to post pull request comments if you want inline insights.

## Troubleshooting

- **Coverage gate fails but pytest passed**: check the aggregated summary. The gate may fail because
  branch coverage dipped even when line coverage is fine.
- **Codecov upload failures**: verify the token (for private repos) and ensure `coverage.xml` exists.
  The workflow will fail fast if the file is missing.
- **Safety or pip-audit failures**: update dependencies or pin vulnerable packages. Use the SARIF
  output from the `Security Scan` workflow for deeper diagnostics.

With these guardrails configured, the CI pipeline enforces consistent quality while giving
contributors actionable feedback when regressions surface.
