# Test Coverage Configuration Guide

This document explains how to configure and use the test coverage workflow (`.github/workflows/ci.yml`) in the TradePulse repository.

## Overview

The CI workflow enforces test coverage requirements on all pull requests and pushes to the `main` branch. It runs tests with coverage reporting and fails the build if coverage drops below the configured threshold.

## Configuration

### Coverage Threshold

The default coverage threshold is **85%**. To change this threshold:

1. Open `.github/workflows/ci.yml`
2. Locate the `COVERAGE_THRESHOLD` environment variable at the top of the file:
   ```yaml
   env:
     COVERAGE_THRESHOLD: "85"
   ```
3. Change the value to your desired threshold (e.g., `"90"` for 90% coverage)

### Coverage Scope

By default, the workflow measures coverage for: `core`, `backtest`, `execution`, and `analytics` packages. To change the coverage scope:

1. Open `.github/workflows/ci.yml`
2. Locate the `COVERAGE_PACKAGES` environment variable at the top of the file:
   ```yaml
   env:
     COVERAGE_PACKAGES: "core,backtest,execution,analytics"
   ```
3. Modify the comma-separated list to target specific packages, for example:
   - `"core,backtest"` - Only core and backtest packages
   - `"tradepulse"` - Single tradepulse package (if using a unified package structure)
   - `"./"` - Entire repository (not recommended for large projects)

You can also configure coverage settings in the `[tool.coverage]` section of `pyproject.toml`.

## Features

The CI workflow includes:

- **Python Version Matrix**: Tests against Python 3.10 and 3.11
- **Parallel Testing**: Uses pytest-xdist with `-n auto` for faster test execution
- **Coverage Reports**: Generates both XML and terminal reports
- **JUnit XML**: Produces JUnit XML reports for test results
- **Artifact Upload**: Uploads both JUnit and coverage XML as GitHub Actions artifacts
- **Codecov Integration**: Automatically uploads coverage data to Codecov with per-version flags
- **Configurable Thresholds**: Easy-to-change coverage threshold via environment variables

## Codecov Integration

### For Public Repositories

Codecov works automatically for public repositories without requiring a token. The workflow will upload coverage reports to Codecov for tracking coverage trends over time.

### For Private Repositories

To enable Codecov for private repositories:

1. Go to [Codecov](https://codecov.io/) and sign in with your GitHub account
2. Add your repository to Codecov
3. Copy the repository upload token from Codecov settings
4. Add the token to your GitHub repository secrets:
   - Go to your repository's **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `CODECOV_TOKEN`
   - Value: Paste the token from Codecov
   - Click **Add secret**

The workflow is configured with `fail_ci_if_error: true`, which means it will fail if Codecov upload fails (useful for catching configuration issues).

## Branch Protection

To make test coverage a required check before merging pull requests:

1. Go to your repository's **Settings** → **Branches**
2. Under "Branch protection rules", click **Add rule** or edit the existing rule for `main`
3. Enter `main` as the branch name pattern (if creating a new rule)
4. Check **Require status checks to pass before merging**
5. Check **Require branches to be up to date before merging**
6. In the search box, find and select:
   - `Test Coverage (Python 3.10)`
   - `Test Coverage (Python 3.11)`
7. (Optional but recommended) Check **Require a pull request before merging**
8. Click **Create** or **Save changes**

This ensures that:
- All tests must pass
- Coverage must meet the threshold
- Both Python versions must succeed
- Pull requests cannot be merged until these checks pass

## Workflow Triggers

The workflow runs automatically on:
- **Pull requests** targeting the `main` branch
- **Pushes** to the `main` branch

This ensures coverage is checked both during code review and after merging.

## Viewing Coverage Reports

### In GitHub Actions

1. Go to the **Actions** tab in your repository
2. Click on a workflow run
3. Expand the "Run tests with coverage" step to see the coverage report in the logs
4. Download JUnit or coverage XML artifacts from the workflow run summary

### On Codecov

1. Visit your repository on [Codecov](https://codecov.io/)
2. View detailed coverage reports, trends, and file-level coverage
3. See coverage changes in pull request comments (automatically posted by Codecov)

## Parallel Test Execution

The workflow uses pytest-xdist with `-n auto` to run tests in parallel, significantly reducing CI time. The number of parallel workers is automatically determined based on available CPU cores.

If you need to disable parallel execution or adjust the number of workers:

1. Open `.github/workflows/ci.yml`
2. Modify the pytest command in the "Run tests with coverage" step:
   - Use `-n 4` to run with 4 workers
   - Use `-n 1` or remove the `-n` flag to disable parallel execution

## Troubleshooting

### Coverage Below Threshold

If the workflow fails due to low coverage:
1. Check which files/functions are not covered in the test output
2. Add tests to cover the missing code
3. Push your changes to re-run the workflow

### Codecov Upload Fails

If Codecov upload fails:
- For private repos: Ensure `CODECOV_TOKEN` secret is correctly configured
- For public repos: Check Codecov service status
- Verify the `coverage.xml` file is being generated correctly

### Python Version Compatibility

The workflow tests against Python 3.10 and 3.11. If your code requires different versions:
1. Edit the `matrix.python-version` in `.github/workflows/ci.yml`
2. Update the branch protection rules to match the new Python versions

### Parallel Test Issues

If tests fail with parallel execution but pass without it:
- Some tests may have race conditions or shared state issues
- Use pytest markers to identify and fix problematic tests
- Consider using `-n 1` or removing `-n auto` as a temporary workaround
