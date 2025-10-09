# Security Shift-Left Program

TradePulse integrates security checks directly into continuous integration so
vulnerabilities are detected before they reach production. This guide explains
how the automated SAST, DAST, and SBOM controls fit together and how to
troubleshoot failures locally.

## Pipeline Overview

| Stage | Tooling | Purpose | CI Job |
| ----- | ------- | ------- | ------ |
| SAST | Bandit, CodeQL, custom secret scanner | Detect insecure patterns in code and leaked secrets. | `secret-scan`, `codeql-analysis` |
| Dependency Security | Safety, pip-audit | Surface vulnerable Python dependencies early. | `dependency-scan` |
| DAST | OWASP ZAP Baseline against the Next.js UI | Probe the running application for runtime weaknesses. | `dast-scan` |
| SBOM | Anchore SBOM Action (Syft) | Produce an SPDX inventory for downstream review. | `sbom` |

The full workflow lives in [`.github/workflows/security.yml`](../.github/workflows/security.yml).

## Static Application Security Testing (SAST)

SAST checks run on every push and pull request:

- **Custom secret scan** reuses
  `core.utils.security.check_for_hardcoded_secrets` to block credential
  leakage.
- **Bandit** scans Python packages under `core/`, `backtest/`, and
  `execution/` in both JSON (artifact) and console output formats.
- **CodeQL** performs language- and framework-aware dataflow analysis using the
  `security-extended` query suite.

### Reproducing SAST locally

```bash
python -c "from core.utils.security import check_for_hardcoded_secrets; check_for_hardcoded_secrets('.')"
bandit -r core backtest execution -ll
# Requires GitHub CLI authentication and the CodeQL CLI
codeql database create codeql-db --language=python --source-root .
codeql database analyze codeql-db --format=sarifv2.1.0 --output=codeql.sarif python-security-extended.qls
```

## Dynamic Application Security Testing (DAST)

The `dast-scan` job deploys the Next.js UI (`apps/web`) to the CI runner and
runs the OWASP ZAP Baseline scan from a container.

- Medium severity findings are printed in the job log for manual triage.
- Unallowlisted high severity findings fail the workflow.
- Known edge-mitigated alerts are tracked in
  [`configs/security/zap-allowlist.json`](../configs/security/zap-allowlist.json).

### Run the DAST scan locally

```bash
npm install --prefix apps/web --no-package-lock
npm run build --prefix apps/web
npx next start --hostname 0.0.0.0 --port 3000 --cwd apps/web &
ZAP_TARGET="http://localhost:3000" zap-baseline.py -t "$ZAP_TARGET" -m 5 -J zap-report.json -x zap-report.xml
kill %1  # stop Next.js once the scan completes (bash job control)
```

Review `zap-report.json` for high severity findings and update the allowlist
only when there is a compensating control documented in production.

## Software Bill of Materials (SBOM)

The `sbom` job uses the Anchore SBOM Action (Syft) to emit an SPDX JSON
inventory of both Python and Node dependencies.

### Generate the SBOM locally

```bash
pip install cyclonedx-bom --quiet
cyclonedx-py --requirements requirements.txt --requirements requirements-dev.txt --format json --output sbom.spdx.json
```

Store SBOM outputs together with build artifacts or feed them into downstream
risk tooling (e.g. vulnerability scanners, license compliance checks).

## Failure Triage Checklist

1. **Identify the failing job** in the GitHub Action run summary.
2. **Download the attached artifacts** (`bandit-report`, `dependency-reports`,
   `zap-dast-report`, or `sbom-spdx`) for detailed evidence.
3. **Confirm whether the issue is already allowlisted** or mitigated by
   compensating controls. Update the documentation if so.
4. **Submit a fix** and rerun the workflow. Security jobs run on every push,
   enabling fast feedback loops.

Treat security gates as blocking issues—exceptions require sign-off from the
security team and a documented remediation plan.
