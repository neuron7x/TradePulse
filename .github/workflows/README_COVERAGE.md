# Test Coverage Configuration Guide

This document explains how to configure and use the test coverage workflow (`.github/workflows/ci.yml`) in the TradePulse repository.

## Overview

The CI workflow enforces test coverage requirements on all pull requests and pushes to the `main` branch. It runs tests with coverage reporting and fails the build if coverage drops below the configured threshold.

## Configuration

### Coverage Threshold

The default coverage threshold is **80%**. To change this threshold:

1. Open `.github/workflows/ci.yml`
2. Locate the `pytest` command in the "Run tests with coverage" step
3. Modify the `--cov-fail-under` parameter:
   ```yaml
   pytest --cov=./ --cov-report=xml --cov-report=term-missing --cov-fail-under=85
   ```
   Replace `80` with your desired threshold (e.g., `85` for 85% coverage)

### Coverage Scope

By default, the workflow measures coverage for the entire repository (`--cov=./`). To measure coverage for specific directories only:

1. Open `.github/workflows/ci.yml`
2. Modify the `--cov` parameter to target specific packages:
   ```yaml
   pytest --cov=core --cov=backtest --cov=execution --cov-report=xml --cov-report=term-missing --cov-fail-under=80
   ```

You can also configure coverage settings in `.coveragerc` file in the repository root.

### Environment Provisioning

The workflow uses [uv](https://github.com/astral-sh/uv) to provision an isolated virtual environment that matches the configured
Python version. The relevant steps in `.github/workflows/ci.yml` look like this:

```yaml
      - name: Set up uv
        uses: astral-sh/setup-uv@v3

      - name: Create virtual environment
        run: uv venv

      - name: Install dependencies
        run: |
          if [ -f requirements.txt ]; then uv pip install -r requirements.txt; fi
          uv pip install pytest pytest-cov

      - name: Run tests with coverage
        run: |
          .venv/bin/python -m pytest --cov=./ --cov-report=xml --cov-report=term-missing --cov-fail-under=80
```

Using uv ensures that dependency resolution respects the selected interpreter (Python 3.11 and 3.12 by default) and avoids
accidentally picking up pre-installed system interpreters that may be incompatible with the declared requirements.

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
   - `Test Coverage (Python 3.11)`
   - `Test Coverage (Python 3.12)`
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

### On Codecov

1. Visit your repository on [Codecov](https://codecov.io/)
2. View detailed coverage reports, trends, and file-level coverage
3. See coverage changes in pull request comments (automatically posted by Codecov)

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

The workflow tests against Python 3.11 and 3.12. If your code requires a different version:
1. Edit the `matrix.python-version` in `.github/workflows/ci.yml`
2. Update the branch protection rules to match the new Python versions
