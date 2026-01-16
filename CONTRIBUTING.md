# Contributing to Databricks Finance Forecasting Platform

Thank you for your interest in contributing! This document outlines our development workflow and standards.

## Git Workflow

We follow the [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) for all contributions.

### Branch Structure

```
main                    # Production-ready code (protected)
├── develop             # Integration branch for features
├── feature/*           # New features
├── bugfix/*            # Bug fixes
├── hotfix/*            # Urgent production fixes
└── release/*           # Release preparation
```

### Branch Naming Convention

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Feature | `feature/<issue-id>-<short-description>` | `feature/42-add-lstm-model` |
| Bug Fix | `bugfix/<issue-id>-<short-description>` | `bugfix/15-fix-date-parsing` |
| Hotfix | `hotfix/<issue-id>-<short-description>` | `hotfix/99-critical-auth-fix` |
| Release | `release/v<version>` | `release/v1.4.0` |

### Workflow Steps

#### 1. Starting a New Feature

```bash
# Ensure you have the latest develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/<issue-id>-<description>

# Work on your feature...
git add .
git commit -m "feat: descriptive commit message"

# Push and create PR
git push -u origin feature/<issue-id>-<description>
```

#### 2. Creating a Pull Request

- All PRs must target `develop` branch (not `main`)
- PRs to `main` only from `release/*` or `hotfix/*` branches
- Link the related issue in PR description
- Ensure all CI checks pass
- Request review from at least one maintainer

#### 3. Release Process

```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.4.0

# Version bump, changelog updates, final testing
# ...

# Merge to main
git checkout main
git merge --no-ff release/v1.4.0
git tag -a v1.4.0 -m "Release v1.4.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.4.0
git push origin develop

# Delete release branch
git branch -d release/v1.4.0
```

#### 4. Hotfix Process

```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/<issue-id>-<description>

# Fix the issue...

# Merge to main AND develop
git checkout main
git merge --no-ff hotfix/<issue-id>-<description>
git tag -a v1.3.1 -m "Hotfix v1.3.1"

git checkout develop
git merge --no-ff hotfix/<issue-id>-<description>
```

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, semicolons, etc.) |
| `refactor` | Code refactoring (no feature/fix) |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Build process, dependencies, etc. |

### Examples

```bash
feat(models): add LSTM forecasting model
fix(api): handle null values in date parsing
docs(readme): update deployment instructions
refactor(preprocessing): extract holiday detection to separate module
test(prophet): add cross-validation unit tests
```

## Code Standards

### Python

- Follow PEP 8 style guide
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for public functions/classes

```python
def train_model(
    data: pd.DataFrame,
    target_col: str,
    horizon: int = 12
) -> Dict[str, Any]:
    """Train a forecasting model on the provided data.

    Args:
        data: Training data with datetime index
        target_col: Name of the target column
        horizon: Number of periods to forecast

    Returns:
        Dictionary containing model metrics and predictions
    """
    ...
```

### TypeScript/React

- Use functional components with hooks
- Prefer TypeScript interfaces over types for objects
- Use meaningful component and variable names

```typescript
interface ForecastChartProps {
  data: ForecastPoint[];
  title: string;
  showConfidenceInterval?: boolean;
}

const ForecastChart: React.FC<ForecastChartProps> = ({
  data,
  title,
  showConfidenceInterval = true
}) => {
  // ...
};
```

## Testing Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain minimum 70% code coverage for new code

### Running Tests

```bash
# Backend tests
pytest backend/tests/ -v

# Frontend tests
npm test
```

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the project style guidelines
- [ ] All tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventional commits
- [ ] PR description clearly explains the changes
- [ ] Related issue is linked

## Issue Guidelines

### Creating Issues

1. Search existing issues first to avoid duplicates
2. Use the appropriate issue template
3. Provide clear reproduction steps for bugs
4. Include environment details (OS, Python/Node version, etc.)

### Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `feature` | New feature request |
| `enhancement` | Improvement to existing feature |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `priority: high` | Critical issues |
| `priority: low` | Nice to have |

## Development Setup

See [README.md](README.md#local-development-guide) for detailed setup instructions.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/debu-sinha/finance-forecast-app.git
cd finance-forecast-app

# Setup environment
./setup-local.sh

# Configure credentials
cp .env.example .env.local
# Edit .env.local with your Databricks credentials

# Start development servers
./start-local.sh
```

## Questions?

- Open a [GitHub Discussion](https://github.com/debu-sinha/finance-forecast-app/discussions) for questions
- Check existing issues and discussions before creating new ones

---

**Thank you for contributing!**
