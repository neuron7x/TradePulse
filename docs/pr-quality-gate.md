# PR Quality Gate Documentation

## Overview

The PR Quality Gate is a comprehensive automated testing and validation system that ensures every pull request meets high standards for code quality, security, and reliability before being merged into the main branch.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pull Request Created                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PR Quality Gate Workflow                       │
│                      (Orchestrator)                              │
└────────┬────────┬────────┬────────┬────────┬────────┬──────────┘
         │        │        │        │        │        │
         ▼        ▼        ▼        ▼        ▼        ▼
    ┌────────┐ ┌────┐ ┌────────┐ ┌─────┐ ┌──────┐ ┌─────────┐
    │  Lint  │ │Test│ │Security│ │Build│ │SBOM  │ │Mutation │
    └────────┘ └────┘ └────────┘ └─────┘ └──────┘ └─────────┘
         │        │        │        │        │          │
         └────────┴────────┴────────┴────────┴──────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Quality Summary Comment     │
         │   Posted on Pull Request      │
         └───────────────────────────────┘
```

## Quality Checks

### 1. Code Quality & Linting (`lint.yml`)

**Purpose**: Ensure code follows consistent style and quality standards

**Checks**:
- **Ruff**: Fast Python linter (checks for code quality issues)
- **Black**: Code formatter (ensures consistent formatting)
- **MyPy**: Type checking (validates type annotations)
- **Shellcheck**: Shell script linting
- **Pre-commit hooks**: Validates all pre-commit hooks pass
- **Slotscheck**: Validates `__slots__` correctness
- **Detect-secrets**: Scans for hardcoded secrets

**Execution Time**: ~3-5 minutes

**Failure Impact**: Blocking - PR cannot be merged

### 2. Comprehensive Testing (`tests.yml`)

**Purpose**: Validate functionality across multiple Python versions

**Test Types**:
- Unit tests (isolated component testing)
- Integration tests (multi-component workflows)
- Property-based tests (Hypothesis-generated test cases)
- E2E smoke tests (end-to-end user workflows)
- Flaky test quarantine (known unstable tests)

**Python Versions**: 3.11, 3.12, 3.13

**Coverage Requirements**:
- Line coverage: ≥97%
- Branch coverage: ≥90%

**Execution Time**: ~15-25 minutes (parallel execution)

**Failure Impact**: Blocking - PR cannot be merged

### 3. Security Scanning (`security.yml`)

**Purpose**: Detect security vulnerabilities and risks

**Scans**:
- **Bandit**: Python security linting
- **Safety**: Known vulnerabilities in dependencies
- **pip-audit**: PyPI vulnerability database
- **CodeQL**: Semantic code analysis
- **Trivy**: Container image scanning
- **Grype**: Alternative container scanner

**Execution Time**: ~10-15 minutes

**Failure Impact**: Blocking for critical vulnerabilities

### 4. Build Verification (`build-wheels.yml`)

**Purpose**: Ensure package builds successfully on all platforms

**Platforms**:
- Ubuntu (latest)
- Windows (latest)
- macOS (latest)

**Python Versions**: 3.11, 3.12, 3.13

**Verification**:
- Wheel creation
- Platform-specific validation (auditwheel, delocate)
- Size and dependency checks

**Execution Time**: ~20-30 minutes (parallel)

**Failure Impact**: Blocking - PR cannot be merged

### 5. SBOM Generation (`sbom.yml`)

**Purpose**: Generate Software Bill of Materials for supply chain security

**Outputs**:
- CycloneDX JSON format
- CycloneDX XML format
- Validated against schema

**Execution Time**: ~3-5 minutes

**Failure Impact**: Blocking - PR cannot be merged

### 6. Mutation Testing (`mutation-tests.yml`)

**Purpose**: Validate test suite effectiveness

**Modes**:
- **PR Mode** (quick): Tests critical modules only (core/agent, core/data, core/metrics)
- **Scheduled Mode** (full): Tests entire codebase

**Execution Time**: 
- PR mode: ~10-15 minutes
- Full mode: ~60-120 minutes

**Failure Impact**: Non-blocking (informational)

### 7. Smoke E2E Tests (`smoke-e2e.yml`)

**Purpose**: Validate end-to-end workflows function correctly

**Modes**:
- **PR Mode** (quick): Reduced dataset, 15-minute timeout
- **Nightly Mode** (full): Full dataset with profiling, 45-minute timeout

**Execution Time**:
- PR mode: ~5-10 minutes
- Full mode: ~30-45 minutes

**Failure Impact**: 
- PR mode: Blocking
- Nightly mode: Alert-only

## Workflow Details

### PR Quality Gate Workflow

The `pr-quality-gate.yml` workflow acts as an orchestrator that:

1. **Monitors** all quality check workflows
2. **Waits** for them to complete
3. **Aggregates** their results
4. **Posts** a comprehensive summary comment on the PR
5. **Fails** if any required check fails

#### Summary Comment Format

```markdown
## 🎯 Quality Gate Summary

### Required Checks

- ✅ **Lint & Code Quality**: Passed
- ✅ **Tests**: Passed
- ✅ **Security Scan**: Passed
- ✅ **Build & Verify Python Wheels**: Passed
- ✅ **CycloneDX SBOM**: Passed
- ✅ **Coverage**: Passed

### Optional Checks

- ✅ **Mutation Tests**: Passed (non-blocking)
- ⚠️ **Nightly Smoke E2E**: Failed (non-blocking)

---
✅ **All required quality checks passed!** This PR is ready for review.
```

## Configuration

### Workflow Triggers

All quality workflows are triggered on:
- `pull_request` targeting `main` or `develop` branches
- Some also run on `push` to `main`/`develop`
- Schedule triggers for nightly/weekly runs

### Concurrency Control

Each workflow uses concurrency groups to:
- Cancel outdated runs when new commits are pushed
- Optimize CI resource usage
- Provide faster feedback

Example:
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Best Practices for Contributors

### Before Opening a PR

1. **Run tests locally**: `make test:all`
2. **Check linting**: `ruff check . && black --check .`
3. **Run type checking**: `mypy core/ backtest/ execution/`
4. **Verify pre-commit hooks**: `pre-commit run --all-files`

### After Opening a PR

1. **Monitor workflows**: Check the Actions tab for your PR
2. **Review quality summary**: Read the automated comment
3. **Fix issues promptly**: Address failures quickly
4. **Request review**: Only after all checks pass

### Common Issues and Solutions

#### Failing Lint Checks

```bash
# Auto-fix most issues
ruff check --fix .
black .

# Run locally to verify
make scripts-lint
```

#### Coverage Too Low

```bash
# Run coverage report
pytest tests/ --cov=core --cov=backtest --cov=execution --cov-branch --cov-report=html

# Open htmlcov/index.html to see uncovered lines
```

#### Security Vulnerabilities

```bash
# Check dependencies
pip-audit
safety check

# Update vulnerable packages
pip install --upgrade <package>
```

#### Build Failures

```bash
# Test build locally
python -m build --sdist --wheel

# Verify wheel
pip install dist/*.whl
```

## Maintenance

### Adding New Quality Checks

1. Create new workflow in `.github/workflows/`
2. Add appropriate triggers (at minimum `pull_request`)
3. Update `pr-quality-gate.yml` to monitor the new workflow
4. Update this documentation

### Adjusting Coverage Thresholds

Edit `.github/workflows/tests.yml`:

```yaml
--cov-fail-under=97  # Line coverage threshold
```

And the branch coverage check in the Python script.

### Optimizing CI Performance

- Use caching for dependencies
- Run expensive checks only on schedule
- Use matrix strategies for parallel execution
- Set appropriate timeouts

## Monitoring and Metrics

### Key Metrics

- **PR Merge Time**: Average time from PR creation to merge
- **Check Failure Rate**: Percentage of PRs that fail initial checks
- **Most Common Failures**: Which checks fail most often
- **CI Resource Usage**: Total minutes used per PR

### GitHub Actions Insights

Navigate to: Repository → Insights → Actions

Monitor:
- Workflow run times
- Success/failure rates
- Resource consumption

## Troubleshooting

### Workflow Not Running

**Symptoms**: PR created but workflows don't start

**Solutions**:
1. Check if workflows are enabled for the repository
2. Verify workflow file syntax (YAML validation)
3. Check branch protection rules
4. Verify GitHub Actions permissions

### Workflow Stuck/Hanging

**Symptoms**: Workflow runs for too long without completing

**Solutions**:
1. Check for infinite loops or blocked operations
2. Review timeout settings
3. Cancel and restart the workflow
4. Check for external service dependencies

### False Positives in Security Scans

**Symptoms**: Security check fails but vulnerability is not applicable

**Solutions**:
1. Review the specific vulnerability
2. Add suppression if justified (with comment explaining why)
3. Update baseline files for secret detection
4. Report false positives to tool maintainers

## Future Enhancements

### Planned Additions

- [ ] Performance regression testing
- [ ] Visual regression testing for UI
- [ ] API contract testing
- [ ] Load testing for critical paths
- [ ] Automated dependency updates with testing
- [ ] AI-powered code review suggestions

### Under Consideration

- Incremental coverage (only check diff)
- Parallel mutation testing
- Custom metrics dashboards
- Integration with issue tracking
- Automated rollback on failures

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [TESTING.md](../TESTING.md) - Detailed testing guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](../SECURITY.md) - Security policies

## Support

For questions or issues with the quality gate:

1. Check this documentation
2. Review workflow logs in the Actions tab
3. Open an issue with the `ci` label
4. Contact the maintainers team

---

**Last Updated**: 2025-10-15  
**Maintained By**: TradePulse Core Team
