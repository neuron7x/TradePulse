<!-- 
GitHub Intervention Protocol (GIP) v1.0
See docs/GIP_SYSTEM_PROMPT.md for complete guidelines
-->

# Context
**What**: <!-- Describe the problem or task -->

**Why**: <!-- Business justification or technical necessity -->

**Type**: <!-- Standard / Hotfix / Infrastructure -->

**Who**: <!-- Responsible persons: @username1, @username2 -->

# Evidence
<!-- Provide concrete proof that justifies this change -->
- Metrics/data that demonstrate the problem or opportunity
- Logs, traces, or error reports
- Monitoring dashboards or profiling results
- Links to discussions, RFCs, or ADRs

# Solution
<!-- Describe your technical solution -->
- Implementation approach
- Alternatives considered and why rejected
- Architecture impact (if any)
- Dependencies or prerequisites

# Test Plan
<!-- Specify concrete tests that validate this change -->
- [ ] **Unit tests**: List specific test files/functions
- [ ] **Integration tests**: Describe workflow tests
- [ ] **E2E tests**: Specify user scenarios covered
- [ ] **Performance tests**: Metrics tracked (if applicable)

**Coverage**: <!-- Current coverage % for affected modules -->

# Rollback Plan
<!-- Step-by-step instructions to revert this change if needed -->
1. <!-- Immediate rollback action (e.g., feature flag, config change) -->
2. <!-- Database migration rollback (if applicable) -->
3. <!-- Time required for full rollback -->
4. <!-- Alternative recovery paths -->

# Documentation Updates
- [ ] **CHANGELOG.md**: Added entry via `newsfragments/` (or mark N/A if docs-only)
- [ ] **ADR**: Created/updated Architecture Decision Record (or mark N/A if not architectural)
- [ ] **README/guides**: Updated user-facing documentation (or mark N/A)
- [ ] **API docs**: Updated API documentation (or mark N/A)

<!-- For hotfix PRs, also include: -->
# Incident & Post-Mortem (Hotfix only)
<!-- Remove this section if not a hotfix -->
- **Incident ticket**: <!-- Link to incident #INC-YYYY-NNN -->
- **Root cause**: <!-- Brief description -->
- **Post-mortem**: <!-- Scheduled date/time or link -->
- **Follow-up tasks**: <!-- Issues created for preventing recurrence -->

---

# Quality Checklist
- [ ] All required sections above are filled out completely
- [ ] Evidence clearly demonstrates need for this change
- [ ] Test plan includes specific test cases with expected outcomes
- [ ] Rollback plan is clear and actionable
- [ ] Documentation is updated or marked N/A with justification
- [ ] Commit messages follow Conventional Commits format
- [ ] CODEOWNERS reviews obtained for affected areas
- [ ] Security, performance, and backward compatibility evaluated
- [ ] CI/CD pipeline passes all checks
