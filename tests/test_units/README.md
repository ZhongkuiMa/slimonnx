# SlimONNX Unit Tests

Fast unit tests for SlimONNX optimization library, designed for CI/CD pipelines.

## Test Structure

```
test_units/
├── test_optimize/          # 7 test files, 63 tests
│   ├── test_mm_add.py      # MatMul+Add fusion (10 tests) ✓
│   ├── test_gemm.py        # Gemm normalization (11 tests) ✓
│   ├── test_redundant.py   # Redundant op removal (14 tests) ✓
│   ├── test_bn_conv.py     # Conv+BN fusion (10 tests)
│   ├── test_bn_gemm.py     # Gemm+BN fusion (5 tests)
│   ├── test_constant_folding.py  # Constant folding (5 tests)
│   └── test_core_api.py    # SlimONNX API (8 tests) ✓
├── test_validate/          # 1 test file, 9 tests
│   └── test_validation.py  # Graph validation (9 tests)
├── test_pattern_detect/    # 1 test file, 6 tests
│   └── test_pattern_detection.py  # Pattern detection (6 tests)
├── test_utils/             # 1 test file, 4 tests
│   └── test_model_utils.py # Model utilities (4 tests)
└── conftest.py            # Shared fixtures and utilities

Total: ~82 unit tests, <5 seconds runtime
```

## Running Tests

```bash
# Fast unit tests with verbose output
pytest tests/test_units/ -v --tb=short -ra

# Single test file
pytest tests/test_units/test_optimize/test_mm_add.py -v

# Specific test function
pytest tests/test_units/test_optimize/test_mm_add.py::TestMatMulAddFusion::test_basic_fusion_success -v

# With coverage
pytest tests/test_units/ --cov=slimonnx --cov-report=term-missing
```

## Test Design Principles

### 1. Broad Code Coverage
- Test public APIs: `SlimONNX.slim()`, `optimize_onnx()`
- Test main optimization paths
- Skip internal helper functions

### 2. Complete Logic Coverage
- Success paths: Happy path optimization
- Failure paths: Conditions that prevent optimization (e.g., rank > 2)
- Edge cases: alpha=0, beta=0, zero padding
- Boundary conditions: Near-zero variance, large matrices

### 3. Expected Error Testing
- Use `pytest.raises()` for ValueError, TypeError
- Test graceful skips (operations not optimized but no error)
- Test error handling for invalid presets, opsets

### 4. Small Fast Inputs
- 2x3 matrices for MatMul/Gemm tests
- 1x3x4x4 tensors for Conv tests
- Fixed (not random) inputs for reproducibility
- No GPU operations

### 5. Fine-Grained Pytest Output
```bash
pytest tests/test_units/ -v --tb=short -ra --showlocals
```
Output shows:
- Each test name with PASSED/FAILED status
- Percentage progress
- Summary of test outcomes
- Detailed tracebacks for failures

### 6. Public API Focus
- Test user-facing methods: `slim()`, `analyze()`, `compare()`
- Test main optimization paths
- Don't test internal functions like `_extract_gemm_params()`

### 7. No Deselected Tests
- NO `@pytest.mark.skip`
- NO `@pytest.mark.skipif`
- All tests must run and pass

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| test_mm_add | 10 | ✓ Complete |
| test_gemm | 11 | ✓ Complete |
| test_redundant | 14 | ✓ Complete |
| test_bn_conv | 10 | - Framework ready |
| test_bn_gemm | 5 | - Framework ready |
| test_constant_folding | 5 | - Framework ready |
| test_core_api | 8 | ✓ Complete |
| test_validation | 9 | - Framework ready |
| test_pattern_detection | 6 | - Framework ready |
| test_model_utils | 4 | - Framework ready |
| **Total** | **82** | **35 Complete** |

## Shared Fixtures

See `conftest.py` for:
- `create_tensor_value_info()` - Create ONNX input/output specs
- `create_initializer()` - Create constant tensors
- `create_minimal_onnx_model()` - Build minimal test models
- `count_ops_by_type()` - Count operation types
- `run_onnx_model()` - Execute models with onnxruntime
- `create_matmul_add_model` - Factory for MatMul+Add patterns
- `create_gemm_model` - Factory for Gemm patterns
- `create_conv_bn_model` - Factory for Conv+BN patterns

## Implementation Notes

### Test File Template
```python
"""Unit tests for [optimization name]."""

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx

from ..conftest import (
    count_ops_by_type,
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
    run_onnx_model,
)


class Test[ClassName]:
    """Test [optimization name] optimization."""

    def test_basic_[feature](self):
        """[What is being tested] - tests [which code branch]."""
        # Setup
        # Optimize
        # Assert
        pass

    def test_[edge_case](self):
        """[Edge case description]."""
        pass
```

### Error Testing Template
```python
def test_invalid_input_raises_error(self):
    """Invalid input should raise ValueError."""
    import pytest

    model = create_invalid_model()
    optimizer = SlimONNX()

    with pytest.raises(ValueError, match="Expected error message"):
        optimizer.slim(model)
```

### Numerical Validation Template
```python
def test_numerical_correctness(self):
    """Optimization preserves outputs."""
    model_before = create_model()
    model_after = optimize_onnx(model_before)

    # Test input
    X_data = np.array([[...]], dtype=np.float32)

    # Run both
    out_before = run_onnx_model(model_before, {"X": X_data})[0]
    out_after = run_onnx_model(model_after, {"X": X_data})[0]

    # Verify
    np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)
```

## GitHub Actions CI/CD

Unit tests run automatically on:
- Push to main/dev branches
- Pull requests

Configuration: `.github/workflows/unit-tests.yml`

Tests must pass on Python 3.11 and 3.12 before merge.
