# Branch Protection Setup Guide

This guide explains how to configure branch protection rules to enforce CODEOWNERS approval requirements for the TradePulse repository.

## Why Branch Protection?

Branch protection rules ensure:
- **Code Quality**: All changes are reviewed before merging
- **Security**: Security-sensitive files require code owner approval
- **Compliance**: Audit trail of all changes to critical code
- **Stability**: Prevent accidental direct pushes to main branch

## Setup Instructions

### Step 1: Navigate to Branch Protection Settings

1. Go to your repository on GitHub
2. Click **Settings** (repository settings, not your account)
3. In the left sidebar, click **Branches**
4. Click **Add branch protection rule** or edit existing rule for `main`

### Step 2: Configure Branch Protection Rule

**Branch name pattern**: `main`

Enable the following settings:

#### Required Reviews
- ✅ **Require a pull request before merging**
  - ✅ **Require approvals**: Set to **1** or more
  - ✅ **Require review from Code Owners**
  - ✅ **Dismiss stale pull request approvals when new commits are pushed**
  - ⚠️ Optional: **Require approval of the most recent reviewable push**

#### Status Checks
- ✅ **Require status checks to pass before merging**
  - ✅ **Require branches to be up to date before merging**
  - Add required status checks:
    - `Secret Scanning (Multi-Tool)`
    - `Bandit Security Linting`
    - `Dependency Vulnerability Scanning`
    - `CodeQL Static Analysis`
    - `tests` (from tests.yml workflow)

#### Additional Protections
- ✅ **Require conversation resolution before merging**
- ✅ **Require signed commits** (optional but recommended)
- ✅ **Require linear history** (optional, prevents merge commits)
- ✅ **Include administrators** (apply rules to repository administrators)
- ❌ **Allow force pushes** (should be disabled)
- ❌ **Allow deletions** (should be disabled)

### Step 3: Save and Test

1. Click **Create** or **Save changes**
2. Test by creating a pull request:
   - Create a branch: `git checkout -b test/branch-protection`
   - Make a change to any file
   - Push and create a PR
   - Verify that code owner approval is required
   - Verify that status checks must pass

## CODEOWNERS File

The `.github/CODEOWNERS` file defines who must approve changes to specific files or directories:

```
# Security-related files require extra scrutiny
/.github/workflows/security.yml @neuron7x
/SECURITY.md @neuron7x
/core/utils/security.py @neuron7x

# Core trading logic
/core/ @neuron7x
/backtest/ @neuron7x
/execution/ @neuron7x
```

## How It Works

1. **Developer creates PR**: Makes changes and opens pull request to `main`
2. **CODEOWNERS notified**: GitHub automatically requests review from code owners
3. **CI checks run**: All security scans and tests execute
4. **Review required**: At least one code owner must approve
5. **Status checks pass**: All required CI jobs must succeed
6. **Merge enabled**: Only then can the PR be merged

## Benefits

### For Security
- All security-sensitive changes reviewed by security-aware maintainers
- Prevents accidental exposure of secrets or vulnerabilities
- Creates audit trail for compliance

### For Code Quality
- Ensures architectural consistency
- Knowledge sharing across team
- Catches bugs before they reach production

### For Compliance
- Clear ownership and accountability
- Documented review process
- Traceable change history

## Troubleshooting

### "Review required from code owner" but no one is listed
- Verify `.github/CODEOWNERS` file exists and has valid syntax
- Check that usernames in CODEOWNERS match GitHub usernames exactly
- Ensure code owner has repository access

### Status checks not appearing
- Verify workflow files are on the `main` branch
- Check that workflow names match exactly in branch protection settings
- Ensure workflows have run at least once

### Can't merge even with approvals
- Check that all required status checks have passed
- Verify branch is up to date with `main`
- Ensure all conversation threads are resolved

## Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [CODEOWNERS Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [Required Status Checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches#require-status-checks-before-merging)

## Support

For questions or issues with branch protection setup:
- Open an issue in the repository
- Contact repository administrators
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
