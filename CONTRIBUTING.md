# Contributing to TradePulse

**Дякуємо за інтерес до TradePulse! / Thank you for your interest in TradePulse!**

This document outlines the rules and processes that enable fast and safe development.

---

## Table of Contents

- [Architectural Framework](#architectural-framework)
- [Prerequisites](#prerequisites)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Pull Request Checklist](#pull-request-checklist)
- [Issue Templates](#issue-templates)
- [Review Process](#review-process)
- [Local Development](#local-development)
- [License](#license)
- [Contact](#contact)

---

## Architectural Framework

TradePulse follows these core principles:

### Contracts-First Design
- **Protocol Buffers**: `.proto` files in `libs/proto/` are the single source of truth for data formats and RPC
- All interfaces must be defined in protobuf before implementation
- Use `buf lint` and `buf generate` to validate and generate code

### Fractal Modular Architecture (FPM-A)
- **Domains**: Organized in `domains/<domain>/<fu>/`
- **Structure**: Each functional unit has:
  - `src/`: Implementation (`core`, `ports`, `adapters`)
  - `tests/`: Unit and integration tests
  - `api/`: Optional API definitions
- **Separation**: Clean boundaries between core logic, ports, and adapters

### Technology Stack
- **Go**: High-performance servers and computation engines
- **Python**: Execution loop, analytics, backtesting
- **Next.js**: Dashboard and visualization
- **Prometheus**: Metrics and monitoring
- **Docker Compose**: Local development and deployment

### Base Interfaces
- **BaseFeature**: Single-purpose transformers for indicators
- **BaseBlock**: Containers for homogeneous features
- All new indicators must implement these interfaces

---

## Prerequisites

### Required Software
- **Python 3.11+**
- **Go 1.22+** (for Go services)
- **Node 18+** (for web dashboard)
- **Docker / Docker Compose**
- **Git**

### Local Setup

```bash
# Clone repository
git clone https://github.com/neuron7x/TradePulse.git
cd TradePulse

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies (dev extras include runtime stack)
pip install -r requirements-dev.lock

# Verify installations
python --version  # Should be 3.11+
go version        # Should be 1.22+
node --version    # Should be 18+

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

### 1. Create a Branch

Follow the naming convention:

```bash
# Feature branch
git checkout -b feat/indicator-awesome-oscillator

# Bug fix
git checkout -b fix/backtest-position-calculation

# Documentation
git checkout -b docs/add-api-examples

# Refactoring
git checkout -b refactor/simplify-kuramoto

# Chore (dependencies, config)
git checkout -b chore/update-dependencies
```

### 2. Update Contracts (if needed)

If your changes affect data structures or APIs:

```bash
# Edit protobuf definitions
vim libs/proto/market/v1/market.proto

# Lint protobuf files
buf lint

# Generate code
buf generate
```

### 3. Implement Logic

- Follow the FPM-A structure
- Implement in appropriate functional unit: `domains/...`
- Separate `core` (business logic), `ports` (interfaces), `adapters` (implementations)
- Add comprehensive docstrings to all public functions

### 4. Write Tests

```bash
# Place tests in the same functional unit
domains/<domain>/<fu>/tests/

# Or in project-wide test directories
tests/unit/
tests/integration/
tests/property/
tests/fuzz/

# Run tests locally
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=backtest --cov=execution --cov=analytics
```

See [TESTING.md](TESTING.md) for detailed testing guidelines.

### 5. Check Code Quality

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
black .

# Type checking
mypy core/ backtest/ execution/

# Security scan
bandit -r core/ backtest/ execution/

# Run all checks
make fpma-check  # Cyclomatic complexity
python -m scripts lint  # Full lint suite
```

### 6. Create Pull Request

- Write a clear description of the problem and solution
- Link to related issues
- Include screenshots for UI changes
- Fill out the PR template completely

---

## Coverage job та Codecov інтеграція

- Всі pull request повинні мати успішний coverage check — це забезпечується workflow `.github/workflows/coverage.yml`.
- Якщо coverage не відображається або з'являється тег "відсутнє покриття":
  1. Переконайтесь, що pipeline не впав і coverage.xml згенеровано.
  2. Перевірте наявність `secrets.CODECOV_TOKEN` у налаштуваннях репозиторію.
  3. Перезапустіть coverage workflow вручну.
  4. Див. інструкцію: https://docs.codecov.com/docs/github-checks
- Badge coverage додається у README автоматично.

**Детальніше:**  
- [Codecov + GitHub інтеграція](https://docs.codecov.com/docs/github-checks)
- [Troubleshooting](https://docs.codecov.com/docs/common-issues)

## Code Standards

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvements
- `test`: Adding or fixing tests
- `chore`: Maintenance tasks, dependency updates

**Examples:**
```
feat(vpin): add streaming VPIN calculator

Implements real-time VPIN calculation using volume buckets.
Closes #123

fix(backtest): correct position sizing calculation

Previously used incorrect risk multiplier.
Fixes #456

docs(readme): add Docker quick start guide

chore(deps): update numpy to 1.26.0
```

### Branch Naming

- `feat/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation
- `refactor/*`: Code refactoring
- `chore/*`: Maintenance
- `test/*`: Test additions/fixes

### Python Style

**Linting and Formatting:**
- Use `ruff` (configured in `pyproject.toml`)
- Follow PEP 8
- Line length: 100 characters
- Use type hints

**Code Example:**
```python
from typing import Optional
import numpy as np

def compute_indicator(
    prices: np.ndarray,
    window: int = 20,
    threshold: Optional[float] = None
) -> tuple[float, dict[str, float]]:
    """Compute custom indicator from price data.
    
    Args:
        prices: 1D array of prices
        window: Lookback window size
        threshold: Optional threshold for signal generation
        
    Returns:
        Tuple of (indicator_value, metadata_dict)
        
    Raises:
        ValueError: If prices array is empty or window is invalid
        
    Example:
        >>> prices = np.array([100, 101, 102, 103])
        >>> value, meta = compute_indicator(prices, window=2)
        >>> print(f"Value: {value:.2f}")
    """
    if len(prices) == 0:
        raise ValueError("prices cannot be empty")
    if window <= 0 or window > len(prices):
        raise ValueError(f"window must be in (0, {len(prices)}]")
    
    # Implementation
    result = np.mean(prices[-window:])
    metadata = {"window": window, "n_prices": len(prices)}
    
    return result, metadata
```

### Go Style

- Use `go fmt` for formatting
- Run `go vet` for static analysis
- Follow standard Go conventions
- Add godoc comments

### TypeScript Style

- Follow Next.js conventions
- Use ESLint (configuration to be added)
- TypeScript strict mode
- Meaningful variable names

### API Stability

- Changes to `.proto` files follow **semantic versioning**
- Document breaking changes in `CHANGELOG.md`
- Provide migration guides for major version bumps

---

## Pull Request Checklist

Before submitting a PR, ensure:

### Code Quality
- [ ] Code follows project style guidelines
- [ ] All linters pass (`ruff`, `mypy`, `bandit`)
- [ ] No hardcoded secrets or credentials
- [ ] Cyclomatic complexity is acceptable (`make fpma-check`)

### Testing
- [ ] New tests added for new functionality
- [ ] All tests pass locally (`pytest tests/`)
- [ ] Coverage maintained or improved
- [ ] Property-based tests added where applicable

### Documentation
- [ ] Public APIs have comprehensive docstrings
- [ ] README.md updated if needed
- [ ] CHANGELOG.md updated (Unreleased section)
- [ ] Comments added for complex logic

### Process
- [ ] Branch name follows convention
- [ ] Commit messages follow Conventional Commits
- [ ] PR description explains problem and solution
- [ ] Related issues linked
- [ ] Screenshots included for UI changes
- [ ] Code of Conduct followed

### Architecture (if applicable)
- [ ] Protocol buffers linted (`buf lint`)
- [ ] Generated code updated (`buf generate`)
- [ ] Dependency graph updated (`make fpma-graph`)
- [ ] No circular dependencies introduced

---

## Issue Templates

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. With data '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- TradePulse version: [e.g., main branch, commit abc123]

**Additional context**
Any other relevant information.

**Logs**
```
Paste relevant logs here
```
```

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other approaches you've thought about.

**Additional context**
Any other relevant information, mockups, examples.

**Proposed Implementation**
If you have ideas about how to implement this.
```

### Documentation Issue Template

```markdown
**Documentation Location**
Which document or section needs improvement?

**Issue**
What is unclear, missing, or incorrect?

**Suggested Fix**
How should it be improved?
```

---

## Review Process

### Review Checklist for Maintainers

**Functionality**
- [ ] Changes solve the stated problem
- [ ] No unintended side effects
- [ ] Edge cases handled

**Code Quality**
- [ ] Code is readable and maintainable
- [ ] Follows project conventions
- [ ] No code smells or anti-patterns
- [ ] Appropriate abstractions

**Security**
- [ ] No security vulnerabilities introduced
- [ ] Input validation present
- [ ] No secrets in code
- [ ] Dependencies are secure

**Testing**
- [ ] Tests are comprehensive
- [ ] Tests actually test the functionality
- [ ] No flaky tests
- [ ] Property-based tests where applicable

**Documentation**
- [ ] Docstrings are clear and complete
- [ ] Examples are provided
- [ ] Public API changes documented

**Performance**
- [ ] No obvious performance issues
- [ ] Algorithms are efficient
- [ ] Memory usage is reasonable

### Review Guidelines

**For Reviewers:**
- Be respectful and constructive
- Explain the "why" behind suggestions
- Distinguish between required changes and suggestions
- Approve when ready, request changes if needed

**For Authors:**
- Respond to all comments
- Don't take feedback personally
- Ask for clarification if needed
- Make requested changes or explain why not

---

## Local Development

### Running Services

```bash
# Start all services (databases, metrics, etc.)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Clean everything
docker compose down -v
```

### Generate Protobuf Code

```bash
# Ensure buf is installed
go install github.com/bufbuild/buf/cmd/buf@latest

# Generate code
python -m scripts gen-proto

# Or manually
buf lint
buf generate
```

### Run Web Dashboard

```bash
cd apps/web
npm install
npm run dev

# Open http://localhost:3000
```

### Run Python CLI

```bash
# Activate virtual environment
source .venv/bin/activate

# Run CLI commands
python -m interfaces.cli analyze --csv sample.csv
python -m interfaces.cli backtest --csv sample.csv
```

### Development Tools

```bash
# Watch for changes and re-run tests
pytest-watch tests/

# Watch TypeScript compilation
cd apps/web && npm run dev

# Watch Go services
# (use air or similar for hot reload)
```

---

## Testing Best Practices

### Unit Tests
- Test one thing at a time
- Use descriptive names
- Mock external dependencies
- Test edge cases and errors

### Integration Tests
- Test realistic workflows
- Use real data when possible
- Test error recovery
- Verify end-to-end behavior

### Property Tests
- Define invariants that must hold
- Use Hypothesis for generation
- Test with diverse inputs
- Catch edge cases automatically

See [TESTING.md](TESTING.md) for complete testing guide.

---

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: type, arg2: type) -> return_type:
    """One-line summary.
    
    Longer description if needed. Explain what the function does,
    any important algorithms, and when to use it.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why this is raised
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

### Documentation Files

- Keep documentation close to code
- Update docs with code changes
- Use examples liberally
- Link between related docs

---

## License and Patents

This project is licensed under **MIT License** (see [LICENSE](LICENSE)).

By contributing, you agree to license your contributions under the same terms.

---

## Contact

**For questions about contributing:**
- Review existing issues and PRs
- Check documentation
- Ask in GitHub Discussions

**For Code of Conduct violations:**
- Email: conduct@tradepulse.local
- See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

**For security issues:**
- Email: security@tradepulse.local
- See [SECURITY.md](SECURITY.md)

---

## Recognition

Contributors are recognized in:
- Release notes
- CHANGELOG.md
- GitHub contributors page

Thank you for contributing to TradePulse! 🚀
