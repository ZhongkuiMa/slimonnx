# SlimONNX

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/slimonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/slimonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/slimonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/slimonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![ONNX 1.16](https://img.shields.io/badge/ONNX-1.16-brightgreen.svg)](https://onnx.ai)
[![ONNXRuntime 1.22](https://img.shields.io/badge/ONNX%20Runtime-1.22-brightgreen.svg)](https://onnxruntime.ai)
[![NumPy 1.26](https://img.shields.io/badge/NumPy-1.26-green.svg)](https://numpy.org/)
[![VNN-COMP 2024](https://img.shields.io/badge/VNN--COMP-2024-orange.svg)](https://sites.google.com/view/vnn2024)

SlimONNX is a pure Python toolkit for optimizing and simplifying ONNX neural network models through graph transformations and operator fusion.

**Extensively tested on all benchmarks from [VNN-COMP 2024](https://sites.google.com/view/vnn2024), covering diverse neural network architectures including feedforward networks, convolutional networks, transformers, and graph neural networks.**

## Motivation

ONNX enables cross-framework model deployment but performs minimal optimization during export. Models exported from frameworks like PyTorch and TensorFlow often contain:

- Redundant operations and identity transformations
- Unfused linear operations that could be combined
- Inconsistent operator representations across ONNX versions
- Complex graph structures that obscure model understanding

SlimONNX addresses these issues through a comprehensive optimization pipeline designed primarily for neural network verification workflows, where simplified models with explicit layer structure are essential for manual inspection and formal analysis.

## Features

- **Pure Python Implementation**: No C/C++ dependencies, simple installation
- **Minimal Dependencies**: Only requires `onnx`, `onnxruntime`, and `numpy`
- **ONNXRuntime Compatible**: Optimized models remain executable with ONNXRuntime
- **Framework Agnostic**: Works with models from any framework that exports to ONNX
- **Composable Optimizations**: Enable specific transformations via configuration
- **Preset Configurations**: Pre-tuned optimization profiles for 23 VNN-COMP 2024 benchmarks
- **Validation Support**: Numerical verification of optimization correctness
- **Analysis Tools**: Model structure inspection and pattern detection
- **Production Ready**: Tested on hundreds of models from VNN-COMP 2024 competition

## Installation

### Requirements

- Python 3.11 or higher
- onnx 1.16.0
- onnxruntime 1.22.0
- numpy 1.26.4

**Important**: Version compatibility matters. Use the specified versions to avoid ONNX opset compatibility issues. Higher versions of onnx/onnxruntime may introduce breaking changes in operator semantics.

### Setup

**Local Installation (Required - No Public PyPI Release)**

SlimONNX is not published to PyPI. Install from the local repository:

```bash
# Clone the repository
git clone https://github.com/ZhongkuiMa/slimonnx.git
cd slimonnx

# Install in editable mode
pip install -e .
```

The `-e` flag installs in "editable" mode, which:
- Creates a link to the source code instead of copying files
- Changes to source code take effect immediately without reinstalling
- Essential for development and testing

**For Contributors (Install with Development Tools):**

```bash
# Install with all development dependencies
# (pytest, ruff, mypy, pre-commit, etc.)
pip install -e ".[dev]"
```

**Verify Installation:**

```bash
python -c "from slimonnx import SlimONNX; print('SlimONNX installed successfully')"
```

### Core Dependencies

```bash
pip install onnx==1.16.0 onnxruntime==1.22.0 numpy==1.26.4
```

### Why These Specific Versions?

- **onnx 1.16.0**: Balanced compatibility with ONNX opset 17-21 (most stable range)
- **onnxruntime 1.22.0**: Matches onnx 1.16.0 for consistent operator behavior
- **numpy 1.26.4**: Required for modern Python 3.11+ compatibility

Using higher versions may cause opset incompatibilities, where optimized models fail to load due to operator definition changes between ONNX versions.

### Optional: Shape Inference Support

For advanced shape inference support, install the companion library:

```bash
pip install shapeonnx
```

## Quick Start

```python
from slimonnx import SlimONNX, get_preset

slimonnx = SlimONNX()

# Optimize with VNN-COMP 2024 benchmark preset
config = get_preset("vit_2023")
slimonnx.slim("model.onnx", "model_optimized.onnx", config=config)

# Or use default optimizations
slimonnx.slim("model.onnx", "model_simplified.onnx")
```

The optimized model can be loaded and executed with ONNXRuntime:

```python
import onnxruntime as ort

session = ort.InferenceSession("model_optimized.onnx")
outputs = session.run(None, {"input": input_data})
```

## Supported Optimizations

SlimONNX implements optimizations across several categories:

### Operator Fusion

- **fuse_matmul_add**: Fuse MatMul+Add into Gemm
- **fuse_gemm_gemm**: Fuse consecutive Gemm operations
- **fuse_gemm_reshape_bn**: Fuse Gemm-Reshape-BatchNormalization
- **fuse_bn_reshape_gemm**: Fuse BatchNormalization-Reshape-Gemm
- **fuse_bn_gemm**: Fuse BatchNormalization-Gemm
- **fuse_transpose_bn_transpose**: Fuse Transpose-BatchNormalization-Transpose
- **fuse_conv_bn**: Fuse Conv-BatchNormalization
- **fuse_bn_conv**: Fuse BatchNormalization-Conv
- **fuse_convtransposed_bn**: Fuse ConvTranspose-BatchNormalization
- **fuse_bn_convtransposed**: Fuse BatchNormalization-ConvTranspose
- **fuse_depthwise_conv_bn**: Fuse depthwise Conv-BatchNormalization
- **fuse_bn_depthwise_conv**: Fuse BatchNormalization-depthwise Conv

### Simplification

- **simplify_conv_to_flatten_gemm**: Convert Conv to Flatten+Gemm where applicable
- **remove_redundant_operations**: Remove identity operations (add zero, multiply one, etc.)
- **constant_folding**: Fold constant expressions into initializers

### Inference Optimizations

- **remove_dropout**: Remove Dropout nodes (enabled by default)

### Graph Transformations

- **simplify_node_name**: Rename nodes sequentially based on topological order
- **reorder_by_strict_topological_order**: Sort nodes in topological order (always applied)
- **simplify_gemm**: Normalize Gemm attributes to canonical form (always applied)

### Always Applied

The following optimizations are always enabled:

- Constants are converted to initializers for shape inference
- Gemm nodes are normalized (alpha=1, beta=1, transA=False, transB=False)
- Graph nodes are topologically sorted

## Architecture

### Design Principles

- **Immutable Configuration**: Frozen dataclass configurations prevent accidental modifications
- **Pure Functional Pipeline**: Model transformations as composable functions
- **Explicit Dependencies**: All optimizations declare their requirements (shapes, batch dimension)
- **Type Safety**: Complete type hints using Python 3.11+ syntax
- **Minimal Abstraction**: Direct operations on ONNX protobuf structures

### Performance Characteristics

- **Single-Pass Optimization**: Most optimizations complete in one graph traversal
- **Lazy Shape Inference**: Shape computation only when required by optimizations
- **Efficient Pattern Matching**: Pre-compiled patterns for common optimization opportunities
- **Topological Ordering**: Ensures correctness of graph transformations

### Module Structure

```
slimonnx/
├── __init__.py                 # Public API exports
├── slimonnx.py                 # Main SlimONNX class
├── configs.py                  # Configuration dataclasses
├── presets.py                  # Preset configurations for benchmarks
├── utils.py                    # Common utilities
├── onnx_attrs.py               # ONNX attribute helpers
├── preprocess/                 # Model preprocessing
│   ├── __init__.py
│   ├── version_converter.py   # ONNX version conversion
│   └── cleanup.py              # Docstring and metadata cleanup
├── optimize_onnx/              # Optimization passes
│   ├── __init__.py
│   ├── _optimize.py            # Main optimization orchestration
│   ├── _cst2initer.py          # Constant to initializer conversion
│   ├── _cst_op.py              # Constant folding
│   ├── _mm_add.py              # MatMul+Add fusion
│   ├── _gemm.py                # Gemm simplification
│   ├── _gemm_gemm.py           # Gemm-Gemm fusion
│   ├── _bn_gemm.py             # BatchNorm-Gemm fusion patterns
│   ├── _bn_transpose.py        # Transpose-BN-Transpose fusion
│   ├── _conv.py                # Conv simplifications
│   ├── _bn_conv.py             # Conv-BN fusion patterns
│   ├── _depthwise_conv.py      # Depthwise Conv-BN fusion
│   ├── _dropout.py             # Dropout removal
│   ├── _redundant.py           # Redundant operation removal
│   ├── _ordering.py            # Topological sorting
│   ├── _name.py                # Node name simplification
│   ├── _utils.py               # Optimization utilities
│   └── constants.py            # ONNX constants and mappings
├── pattern_detect/             # Pattern detection for analysis
│   ├── __init__.py
│   ├── registry.py             # Pattern registry
│   ├── matmul_add.py           # MatMul+Add patterns
│   ├── gemm_chains.py          # Gemm chain patterns
│   ├── gemm_bn.py              # Gemm-BN patterns
│   ├── transpose_bn.py         # Transpose-BN patterns
│   ├── conv_bn.py              # Conv-BN patterns
│   ├── depthwise_conv.py       # Depthwise Conv patterns
│   ├── constant_ops.py         # Constant operation patterns
│   ├── redundant_ops.py        # Redundant operation patterns
│   ├── reshape_chains.py       # Reshape chain patterns
│   └── dropout.py              # Dropout patterns
├── model_validate/             # Model validation
│   ├── __init__.py
│   ├── onnx_checker.py         # ONNX checker validation
│   ├── runtime_validator.py    # ONNXRuntime validation
│   ├── graph_validator.py      # Graph structure validation
│   └── numerical_compare.py    # Numerical output comparison
└── structure_analysis/         # Model structure analysis
    ├── __init__.py
    ├── analyzer.py             # Structure analyzer
    ├── topology.py             # Topology analysis
    └── reporter.py             # JSON report generation
```

### Optimization Pipeline

```
Input ONNX Model
    │
    ├─> Preprocessing
    │   ├─> Load model
    │   ├─> Version conversion (target opset)
    │   ├─> Shape inference
    │   └─> Clear docstrings
    │
    ├─> Optimization Passes (configurable)
    │   ├─> Constant to initializer (always)
    │   ├─> Remove dropout
    │   ├─> Constant folding
    │   ├─> MatMul+Add → Gemm
    │   ├─> Gemm simplification (always)
    │   ├─> Gemm-Gemm fusion
    │   ├─> BatchNorm-Gemm fusion
    │   ├─> Transpose-BN-Transpose fusion
    │   ├─> Conv-BN fusion
    │   ├─> Depthwise Conv-BN fusion
    │   ├─> Conv to Flatten+Gemm
    │   ├─> Remove redundant operations
    │   ├─> Topological reordering (always)
    │   └─> Node name simplification
    │
    ├─> Validation (optional)
    │   ├─> ONNX checker
    │   ├─> ONNXRuntime loading
    │   └─> Numerical comparison
    │
    └─> Save Optimized Model
```

## Usage

### Basic Example

```python
from slimonnx import SlimONNX, OptimizationConfig

slimonnx = SlimONNX()

# Default optimization (only always-applied transformations)
slimonnx.slim(
    "model.onnx",
    "model_simplified.onnx",
)

# Custom optimization configuration
config = OptimizationConfig(
    fuse_matmul_add=True,
    fuse_gemm_gemm=True,
    remove_redundant_operations=True,
)

slimonnx.slim(
    "model.onnx",
    "model_optimized.onnx",
    config=config,
)
```

### Using Presets

SlimONNX provides pre-tuned configurations for common benchmarks:

```python
from slimonnx import SlimONNX, get_preset

slimonnx = SlimONNX()

# Use preset for specific benchmark
config = get_preset("vit_2023")

slimonnx.slim(
    "vit_model.onnx",
    "vit_model_optimized.onnx",
    config=config,
)

# Enable all optimizations
from slimonnx import all_optimizations

config = all_optimizations(has_batch_dim=True)

slimonnx.slim(
    "model.onnx",
    "model_fully_optimized.onnx",
    config=config,
)
```

Available presets: `acasxu_2023`, `vit_2023`, `cgan_2023`, `cifar100_2024`, `nn4sys_2023`, and more. See `slimonnx/presets.py` for the complete list.

### Validation

Verify that optimization preserves model outputs:

```python
from slimonnx import SlimONNX, OptimizationConfig, ValidationConfig

slimonnx = SlimONNX()

opt_config = OptimizationConfig(
    fuse_conv_bn=True,
    fuse_matmul_add=True,
)

val_config = ValidationConfig(
    validate_outputs=True,
    num_samples=10,
    rtol=1e-5,
    atol=1e-6,
)

result = slimonnx.slim(
    "model.onnx",
    "model_optimized.onnx",
    config=opt_config,
    validation=val_config,
)

print(f"Node reduction: {result['reduction']} ({result['reduction_pct']:.1f}%)")
print(f"Validation: {result['validation']['all_match']}")
```

### Analysis

Analyze model structure and detect optimization opportunities:

```python
from slimonnx import SlimONNX

slimonnx = SlimONNX()

# Analyze model
report = slimonnx.analyze("model.onnx")

print(f"Total nodes: {report['structure']['node_count']}")
print(f"Input count: {report['structure']['input_count']}")
print(f"Output count: {report['structure']['output_count']}")
print(f"Fusible patterns: {report['recommendations']['fusible_patterns']}")
print(f"Redundant patterns: {report['recommendations']['redundant_patterns']}")

# Detect specific patterns
patterns = slimonnx.detect_patterns("model.onnx")
for pattern_name, info in patterns.items():
    if info['count'] > 0:
        print(f"{pattern_name}: {info['count']} occurrences")
```

### Model Comparison

Compare original and optimized models:

```python
from slimonnx import SlimONNX

slimonnx = SlimONNX()

comparison = slimonnx.compare(
    "model_original.onnx",
    "model_optimized.onnx",
)

print(f"Node reduction: {comparison['diff']['nodes']['reduction']}")
print(f"Patterns fixed: {len(comparison['diff']['patterns_fixed'])}")
```

## API Reference

### SlimONNX Class

Main class providing optimization and analysis methods.

#### slim()

Optimize ONNX model.

**Parameters**:
- `onnx_path` (str): Path to input ONNX model
- `target_path` (str | None): Path to save optimized model (default: {input}_simplified.onnx)
- `config` (OptimizationConfig | None): Optimization configuration
- `validation` (ValidationConfig | None): Validation configuration

**Returns**: dict | None - Optimization report if validation enabled, else None

#### analyze()

Analyze model structure and detect patterns.

**Parameters**:
- `onnx_path` (str): Path to ONNX model
- `config` (OptimizationConfig | None): Configuration for has_batch_dim
- `analysis` (AnalysisConfig | None): Analysis configuration

**Returns**: dict - Comprehensive analysis report

#### compare()

Compare two ONNX models.

**Parameters**:
- `original_path` (str): Path to original model
- `optimized_path` (str): Path to optimized model

**Returns**: dict - Comparison report

#### validate()

Validate model correctness.

**Parameters**:
- `onnx_path` (str): Path to ONNX model
- `config` (OptimizationConfig | None): Configuration for has_batch_dim

**Returns**: dict - Validation report

#### detect_patterns()

Detect optimization patterns.

**Parameters**:
- `onnx_path` (str): Path to ONNX model
- `config` (OptimizationConfig | None): Configuration for has_batch_dim

**Returns**: dict - Pattern detection report

### Configuration Classes

All configuration classes are immutable frozen dataclasses.

#### OptimizationConfig

Controls which optimizations to apply.

**Key Parameters**:
- `fuse_matmul_add` (bool): Fuse MatMul+Add to Gemm (default: False)
- `fuse_conv_bn` (bool): Fuse Conv+BatchNorm (default: False)
- `constant_folding` (bool): Fold constant operations (default: False)
- `remove_redundant_operations` (bool): Remove no-op nodes (default: False)
- `simplify_node_name` (bool): Rename nodes sequentially (default: False)
- `has_batch_dim` (bool): Model has batch dimension (default: True)

See `slimonnx/configs.py` for complete parameter list.

#### ValidationConfig

Controls output validation.

**Parameters**:
- `validate_outputs` (bool): Enable validation (default: False)
- `num_samples` (int): Number of test samples (default: 5)
- `rtol` (float): Relative tolerance (default: 1e-5)
- `atol` (float): Absolute tolerance (default: 1e-6)
- `input_bounds` (tuple | None): Input value bounds
- `test_data_path` (str | None): Path to test data

#### AnalysisConfig

Controls analysis exports.

**Parameters**:
- `export_json` (bool): Export analysis JSON (default: False)
- `json_path` (str | None): JSON export path
- `export_topology` (bool): Export topology JSON (default: False)
- `topology_path` (str | None): Topology export path

## Testing

SlimONNX includes comprehensive test suites:

### Run All Tests

```bash
cd slimonnx
python -m pytest tests/
```

### Test Preprocessing

```bash
python -m pytest tests/test_validation.py --preprocess-only
```

### Test Validation

```bash
python -m pytest tests/test_validation.py --validate-only
```

### Optimize Benchmarks

```bash
python -m pytest tests/test_benchmarks.py --optimize
```

### Verify Against Baseline

```bash
python -m pytest tests/test_benchmarks.py --verify
```

## ONNX Version Compatibility

- **Recommended Opset**: 20
- **Maximum Tested**: 21
- **Minimum Tested**: 17

Models are automatically converted to target opset during preprocessing. Use `onnx.version_converter` for manual version conversion.

### ONNXRuntime Compatibility

All optimizations preserve ONNXRuntime compatibility. Optimized models can be executed with onnxruntime 1.22.0.

**Validation**: Every optimization is tested against ONNXRuntime to ensure numerical equivalence between original and optimized models. The test suite includes:

- Output validation with random inputs
- Numerical tolerance checks (rtol=1e-5, atol=1e-6)
- Comparison against baseline models from VNN-COMP 2024

**Note**: While optimized models are ONNXRuntime compatible, version mismatches between onnx and onnxruntime may cause loading failures. Always use matching versions as specified in the installation section.

## Testing and Validation

SlimONNX has been extensively tested on the complete VNN-COMP 2024 benchmark suite:

### VNN-COMP 2024 Benchmarks Coverage

All 23 benchmarks from the International Verification of Neural Networks Competition 2024:

- acasxu_2023
- cctsdb_yolo_2023
- cersyve
- cgan_2023
- cifar100_2024
- collins_aerospace_benchmark
- collins_rul_cnn_2022
- cora_2024
- dist_shift_2023
- linearizenn
- lsnc
- lsnc_relu
- malbeware
- metaroom_2023
- ml4acopf_2024
- nn4sys_2023
- relusplitter
- safenlp_2024
- sat_relu
- soundnessbench
- tinyimagenet_2024
- tllverifybench_2023
- traffic_signs_recognition_2023
- vggnet16_2022
- vit_2023
- yolo_2023

### Test Results

- **Total Models Tested**: 100+ models across all benchmarks
- **Optimization Success Rate**: 100% (all models successfully optimized)
- **ONNXRuntime Compatibility**: 100% (all optimized models loadable and executable)
- **Numerical Validation**: Validated on models with test data (safenlp, cgan, vit, etc.)

Each benchmark has a tuned preset configuration in `slimonnx/presets.py` optimized for its specific architecture patterns.

## Known Limitations

- Shape inference requires models with explicit tensor shapes or batch dimension information
- Some optimizations (like constant folding) require successful shape inference
- Models with dynamic shapes may have limited optimization opportunities
- Batch normalization fusion assumes inference mode (training=False)

## Related Projects

- **[ShapeONNX](https://github.com/ZhongkuiMa/shapeonnx)**: Advanced shape inference for ONNX models. SlimONNX uses ShapeONNX for shape-dependent optimizations.
- **[TorchVNNLIB](https://github.com/ZhongkuiMa/torchvnnlib)**: PyTorch library for neural network verification. Often used in conjunction with SlimONNX for model verification tasks. This convert the VNNLIB data files to `.pth` format for PyTorch or `.npz` format for NumPy.
- **[VNN-COMP](https://sites.google.com/view/vnn2024)**: International Verification of Neural Networks Competition. SlimONNX is tested on all VNN-COMP 2024 benchmarks.
- **[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)**: Alternative ONNX optimization tool with different optimization strategies.

## Development & Testing

### Local Development Setup

1. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

### Code Quality Standards

All code must pass these checks before committing:

**Linting:**
```bash
ruff check src/slimonnx tests
```

**Formatting:**
```bash
ruff format src/slimonnx tests
```

**Type checking:**
```bash
mypy src/slimonnx
```

**Running Tests:**
```bash
pytest tests/ --cov=slimonnx --cov-report=term-missing
```

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

## Contributing

Contributions are welcome! Please follow this workflow:

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

### Getting Help

- **Questions?** Open an issue or ask in the pull request discussion
- **Bug report?** Open an issue with reproduction steps and environment details
- **Security issue?** Please email maintainers directly (do not open public issue)

## License

MIT License. See LICENSE file for details.
