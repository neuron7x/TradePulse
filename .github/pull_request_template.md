# Summary
- [ ] Explain the purpose of this change and the primary outcomes.
- [ ] Link related issues, follow-up tasks, or design docs.

# Testing
- [ ] `pytest` (unit, integration, property, fuzz, contracts, security)
- [ ] Coverage meets CI gate (`--cov-branch --cov-fail-under=90`; Codecov project + patch checks green)
- [ ] Data quality gates (`pytest tests/data` or `python scripts/data_sanity.py ...`)
- [ ] Contract compatibility (`pytest tests/contracts`)
- [ ] Security scans (Bandit, secret-leak detection)
- [ ] UI smoke & accessibility (Playwright + aXe)

# Quality Checklist
- [ ] Documentation updated or confirmed not required.
- [ ] Telemetry/metrics reviewed or updated.
- [ ] Security, performance, and backward compatibility risks evaluated.
- [ ] CODEOWNERS for touched areas acknowledged.
