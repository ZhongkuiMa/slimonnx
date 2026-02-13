# Contributing to SlimONNX

Contributions are welcome! Please follow this workflow to ensure smooth collaboration.

## Development Setup

### Local Development Setup

1. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Code Quality Standards

All code must pass these checks before committing:

### Linting

```bash
ruff check src/slimonnx tests
```

### Formatting

```bash
ruff format src/slimonnx tests
```

### Type Checking

```bash
mypy src/slimonnx
```

### Running Tests

```bash
pytest tests/ --cov=slimonnx --cov-report=term-missing
```

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/slimonnx.git
cd slimonnx
```

### 2. Set Up Development Environment

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Branch naming conventions:
# - feature/feature-name  (new functionality)
# - fix/bug-description   (bug fixes)
# - docs/documentation-update (documentation only)
# - refactor/refactor-description (code refactoring)
# - test/test-description (test additions)
```

### 4. Make Changes and Test

```bash
# Make your code changes

# Format code
ruff format src/slimonnx tests

# Check for issues
ruff check src/slimonnx tests

# Run type checking
mypy src/slimonnx

# Run tests
pytest tests/ --cov=slimonnx --cov-report=term-missing

# Verify all checks pass
pre-commit run --all-files
```

### 5. Commit Your Changes

```bash
# Stage your changes
git add <files>

# Commit with descriptive message
git commit -m "type: Brief description of changes

Detailed explanation of what changed and why.
Include context and any relevant issue numbers.

Fixes #123"
```

**Commit message format:**
- **Type prefixes:** `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- **First line:** Brief summary (50 characters max)
- **Body:** Detailed explanation (wrap at 72 characters)
- **Footer:** Reference issues/PRs (`Fixes #123` or `Closes #456`)

### 6. Push to Your Fork

```bash
# Push your feature branch to your fork
git push origin feature/your-feature-name

# If you need to update an existing PR:
git push origin feature/your-feature-name --force-with-lease
```

### 7. Open a Pull Request

1. **Go to the original repository** on GitHub
2. **Click "New Pull Request"** or "Compare & pull request"
3. **Select your fork and feature branch** as the source
4. **Fill out the PR template:**
   - **Title:** Clear, concise description (same as commit title)
   - **Description:** What changed and why
   - **Reference issues:** `Fixes #123` or `Closes #456`
   - **Breaking changes:** List any (if applicable)
   - **Testing:** Describe how to test the changes
   - **Checklist:** Confirm tests added, docs updated, etc.

## Pull Request Guidelines

### Pull Request Requirements

Before your PR can be merged, it must:

- ✅ **Pass all CI checks** (ruff linting, ruff formatting, pytest on Python 3.11 and 3.12)
- ✅ **Maintain or improve code coverage** (currently 96%, minimum 95%)
- ✅ **Include tests** for new functionality or bug fixes
- ✅ **Update documentation** for API changes or new features
- ✅ **Follow code style** (enforced by ruff and pre-commit hooks)
- ✅ **Have a clear description** explaining the changes

### Code Review Process

1. **Automated checks run first** - GitHub Actions runs linting and tests
2. **Maintainer review** - At least one maintainer will review
3. **Address feedback** - Make requested changes and push updates
4. **Approval and merge** - Once approved and all checks pass, maintainer will merge

### After Your PR is Merged

```bash
# Switch back to main branch
git checkout main

# Update your local main branch
git pull upstream main

# Delete your local feature branch (optional)
git branch -d feature/your-feature-name

# Delete your remote feature branch (optional)
git push origin --delete feature/your-feature-name
```

### Direct Push to Main (Restricted)

⚠️ **Direct pushes to the `main` branch are not allowed.** All changes must go through the pull request process to ensure code review and CI validation.

## Testing

### Testing Details

- **Unit Test Count**: 1,060 comprehensive unit tests
- **Benchmark Test Count**: 46 benchmark tests (316 additional skipped - require test data)
- **Test Coverage**: 96% (2,690/2,790 statements)
- **Test Execution Time**: 1.70 seconds (unit tests), 0.59 seconds (benchmark tests)
- **Test Status**: ✅ All 1,060 unit tests passing, 46 benchmark tests passing
- **Python Versions Tested**: 3.11, 3.12
- **Test Structure**:
  - `tests/test_units/`: 70+ test files organized by module
  - `tests/test_benchmarks/`: Comprehensive benchmark tests including:
    - VNN-COMP 2024 preset validation (44 tests)
    - Basic optimization tests (2 tests)
    - Regression tests (270+ skipped - require baseline data)
    - Analysis and validation tests (require large model processing)
- **Run All Unit Tests**: `pytest tests/test_units/`
- **Run All Benchmark Tests**: `pytest tests/test_benchmarks/test_vnncomp2024_benchmarks.py tests/test_benchmarks/test_basic.py`
- **Run Specific Tests**: `pytest tests/test_units/test_optimize/ -v`
- **View Coverage**: After running tests with coverage, open `htmlcov/index.html`

### Latest Test Results

**Last Run:** January 3, 2026

```
✅ All Tests Passing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit Tests:        1,060 passed
Benchmark Tests:   46 passed, 270 skipped
Total:             1,106 executed, 270 skipped
Coverage:          96% (2,690/2,790 statements)
Duration:          1.70s (unit tests) + 0.59s (benchmarks)
Python:            3.11, 3.12
Warnings:          6 (non-critical, test assertions)
```

**Test Breakdown:**
- **Unit Tests** (1,060/1,060 - 100% pass rate)
  - Optimization tests: 300+ tests
  - Pattern detection tests: 200+ tests
  - Model validation tests: 150+ tests
  - Structure analysis tests: 100+ tests
  - Other modules: 310+ tests

- **Benchmark Tests** (46/46 - 100% pass rate)
  - VNN-COMP 2024 preset validation: 44 tests (all benchmarks configured correctly)
  - Basic optimization tests: 2 tests (Conv-BN fusion, basic optimization)

**Coverage by Module:**
- `slimonnx.py`: 99% (115/116 statements)
- `utils.py`: 99% (130/131 statements)
- `pattern_detect/`: 94-100% (registry, constant_ops at 100%)
- `optimize_onnx/`: 89-100% (depthwise_conv, dropout, gemm, mm_add at 100%)
- `model_validate/`: 88-100% (onnx_checker at 100%)
- `structure_analysis/`: 100% (all modules)
- `presets.py`: 100%

**Coverage Gaps (4% coverage):**
- Primarily edge cases and error paths in constant folding (`_cst_op.py`)
- Pattern detection for unsupported patterns (`reshape_negative_one.py`)
- Version converter warnings for out-of-range opsets
- Non-critical runtime error handling paths

## CI/CD

### GitHub Actions CI/CD

The project uses GitHub Actions for continuous integration:

- **Triggers**:
  - Every push to `main` branch
  - Every pull request to `main` branch
  - Scheduled daily at 8 AM UTC
  - Manual dispatch via GitHub UI
- **Python Versions**: 3.11 and 3.12
- **Checks**:
  - Ruff linting and formatting
  - Pytest on both Python versions
  - Coverage measurement and Codecov upload (Python 3.11 only)

**Note**: Type checking (mypy) is enforced via pre-commit hooks but not currently in CI.

## Getting Help

- **Questions?** Open an issue or ask in the pull request discussion
- **Bug report?** Open an issue with reproduction steps and environment details
- **Security issue?** Please email maintainers directly (do not open public issue)
