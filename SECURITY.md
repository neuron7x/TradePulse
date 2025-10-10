# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Responsible Disclosure Policy

We strongly encourage responsible disclosure and commit to working with researchers to promptly address reported issues.

- **Primary contact**: `security@tradepulse.local`
- **Backup contact**: Direct message the maintainers via the [GitHub Security Advisory](https://github.com/neuron7x/TradePulse/security/advisories/new) form.
- **SLA overview**:
  - Acknowledge receipt within **48 hours**.
  - Provide an initial assessment within **5 business days**.
  - Deliver fix timelines based on severity (see below) and share remediation status updates at least every **5 business days** until resolution.
- **Safe harbor**: Good-faith security research that complies with this policy will not be subject to legal action or revocation of access.

### Disclosure Process

1. **Report**: Send details to **security@tradepulse.local** or via [GitHub Security Advisories](https://github.com/neuron7x/TradePulse/security/advisories/new)
   - Include description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)
   - Proof of concept (if available)
   - Your contact information

2. **Acknowledgment**: We will acknowledge receipt within 48 hours

3. **Investigation**: We will investigate and provide updates every 5 business days
   - Initial triage: 24-48 hours
   - Impact assessment: 3-5 business days
   - Regular status updates

4. **Resolution**: 
   - Critical vulnerabilities: Fixed within 7 days
   - High severity: Fixed within 30 days
   - Medium/Low severity: Fixed within 90 days
   - You will be credited in release notes (unless you prefer to remain anonymous)

5. **Disclosure**: After a fix is released, we will publish a security advisory
   - CVE assignment if applicable
   - Public disclosure 7 days after patch release
   - Coordinated with reporter

### Severity Classification

| Severity | Description | Response Time |
|----------|-------------|---------------|
| Critical | Remote code execution, authentication bypass | 7 days |
| High | Data exposure, privilege escalation | 30 days |
| Medium | DoS, information disclosure | 90 days |
| Low | Minor issues with minimal impact | 180 days |

### What to Report

We're interested in any type of security issue, including:
- Authentication/authorization bypasses
- Data exposure or leakage
- Injection vulnerabilities (SQL, command, code injection)
- Cross-site scripting (XSS) in web interfaces
- Denial of service vulnerabilities
- Cryptographic issues
- Dependency vulnerabilities with known exploits
- API security issues
- Secrets or credentials in code
- Insecure defaults or configurations

### What NOT to Report

The following are **not** considered security vulnerabilities:
- Vulnerabilities in dependencies without a proof of exploit
- Missing security headers without demonstrated impact
- Issues requiring physical access
- Social engineering attacks
- Missing best practices without security impact
- Issues in deprecated or EOL dependencies

### Bug Bounty Program

We currently do not offer monetary rewards for security findings. However, we recognize and credit all valid security reports publicly (with permission).

### Hall of Fame

We maintain a [Security Hall of Fame](SECURITY_HALL_OF_FAME.md) recognizing security researchers who have responsibly disclosed vulnerabilities.

---

## Security Best Practices

### For Contributors

#### 1. Secrets Management

**Never commit secrets to the repository:**
- API keys
- Passwords
- Private keys
- Connection strings
- Tokens

**Use environment variables:**
```bash
# Good
export TRADING_API_KEY="your-key-here"

# Bad (never do this)
api_key = "sk-12345..."  # in code
```

**Check for secrets before committing:**
```bash
# Use git-secrets or similar tools
git secrets --scan
```

#### 2. Input Validation

**Always validate and sanitize inputs:**
```python
# Good
def process_price(price: float) -> float:
    if not isinstance(price, (int, float)):
        raise ValueError("Price must be numeric")
    if price <= 0:
        raise ValueError("Price must be positive")
    return float(price)

# Bad
def process_price(price):
    return float(price)  # No validation
```

#### 3. Dependency Management

**Keep dependencies up to date:**
```bash
# Regular updates
pip install -U -r requirements.txt

# Check for known vulnerabilities
pip-audit

# Or use safety
safety check
```

**Review dependency changes:**
- Check changelogs before updating
- Test thoroughly after updates
- Pin versions in production

**CycloneDX SBOM generation:**
- Every push, pull request, and release automatically generates validated CycloneDX SBOMs (JSON and XML).
- Download SBOM artifacts from the `CycloneDX SBOM` workflow run or from the published release assets.
- Use these SBOMs to audit dependency inventories and share with stakeholders.

**Container signing & provenance:**
- Release container images are pushed to GHCR with keyless [Sigstore Cosign](https://github.com/sigstore/cosign) signatures attached to the digest.
- Every release produces a SLSA v3 provenance statement for the container image via the official [slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator).
- Consumers can verify signatures with `cosign verify ghcr.io/<owner>/<repo>@<digest>` and download provenance attestations directly from GHCR.

#### 4. Code Review

**Security checklist for PRs:**
- [ ] No hardcoded secrets
- [ ] Input validation on all external data
- [ ] Proper error handling (no sensitive data in errors)
- [ ] Authentication/authorization checks
- [ ] SQL queries use parameterization
- [ ] File operations validate paths
- [ ] External commands properly escaped

---

## Security Tooling

### Automated Security Scanning

We use multiple tools in CI/CD:

#### 1. CodeQL (GitHub Advanced Security)
```yaml
# Automatically scans code for vulnerabilities
# Configured in .github/workflows/codeql.yml
```

#### 2. Bandit (Python Security Linter)
```bash
# Run locally
bandit -r core/ backtest/ execution/ interfaces/

# Common issues detected:
# - Use of assert in production
# - Hardcoded passwords
# - SQL injection risks
# - Shell injection risks
```

#### 3. Safety (Dependency Checker)
```bash
# Check for known vulnerabilities
safety check --json

# Update requirements
safety check --update
```

#### 4. pip-audit
```bash
# Audit Python packages
pip-audit --desc --format json
```

#### 5. Semgrep (Static Analysis)
```bash
# Run semantic code analysis
semgrep --config=auto .
```

### Pre-commit Hooks

Install security checks as pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# .pre-commit-config.yaml should include:
# - bandit
# - detect-secrets
# - check-added-large-files
```

---

## Common Security Patterns

### 1. API Key Management

**Development:**
```bash
# .env (never commit this file)
EXCHANGE_API_KEY=your-key
EXCHANGE_API_SECRET=your-secret
```

**Production:**
- Use environment variables
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Rotate keys regularly

### 2. Database Queries

**Use parameterized queries:**
```python
# Good
cursor.execute("SELECT * FROM trades WHERE symbol = ?", (symbol,))

# Bad
cursor.execute(f"SELECT * FROM trades WHERE symbol = '{symbol}'")
```

### 3. File Operations

**Validate file paths:**
```python
import os

def read_config(filename: str) -> dict:
    # Prevent path traversal
    base_dir = "/app/configs"
    full_path = os.path.normpath(os.path.join(base_dir, filename))
    
    if not full_path.startswith(base_dir):
        raise ValueError("Invalid file path")
    
    with open(full_path, 'r') as f:
        return json.load(f)
```

### 4. Error Handling

**Don't expose sensitive information:**
```python
# Good
try:
    result = execute_trade(order)
except Exception as e:
    logger.error(f"Trade execution failed: {type(e).__name__}")
    return {"error": "Trade execution failed"}

# Bad
except Exception as e:
    return {"error": str(e)}  # May expose database structure, API keys, etc.
```

---

## Security Checklist for Releases

Before releasing a new version:

- [ ] Run all security scanners (bandit, safety, semgrep)
- [ ] Update all dependencies to latest secure versions
- [ ] Review and rotate any compromised credentials
- [ ] Check for hardcoded secrets in codebase
- [ ] Verify authentication/authorization logic
- [ ] Review recent security advisories for dependencies
- [ ] Update CHANGELOG.md with security fixes
- [ ] Create security advisory if needed

---

## Threat Model

### Assets
- Trading strategies and algorithms
- Market data and analytics
- API credentials and secrets
- User funds and positions
- System availability

### Threats
- **Unauthorized Access**: Compromise of API keys or credentials
- **Data Manipulation**: Tampering with market data or orders
- **Information Disclosure**: Leakage of trading strategies
- **Denial of Service**: System unavailability during critical trading
- **Supply Chain**: Compromised dependencies

### Mitigations
- Environment-based secrets management
- Input validation and sanitization
- Encrypted communications (TLS/SSL)
- Rate limiting and monitoring
- Regular security audits
- Dependency scanning

---

## Incident Response

In case of a security incident:

1. **Contain**: Isolate affected systems
2. **Assess**: Determine scope and impact
3. **Notify**: Contact security@tradepulse.local
4. **Remediate**: Apply fixes and patches
5. **Review**: Post-mortem and lessons learned
6. **Disclose**: Responsible disclosure to users

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security.html)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

## Contact

- **Security Issues**: security@tradepulse.local
- **General Issues**: [GitHub Issues](https://github.com/neuron7x/TradePulse/issues)

---

**Last Updated**: 2025-01-01
