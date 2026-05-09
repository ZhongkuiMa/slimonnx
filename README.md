# SlimONNX

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/slimonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/slimonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/slimonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/slimonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Optimize ONNX models through operator fusion, constant folding, and redundant operation removal for neural network verification workflows.

## Installation

```bash
git clone https://github.com/ZhongkuiMa/slimonnx.git
cd slimonnx
pip install -e ".[dev]"
```

Requirements: Python 3.11+, onnx, onnxruntime, numpy, shapeonnx

## Quick Start

```python
from slimonnx import SlimONNX, OptimizationConfig

slimonnx = SlimONNX()

# Default optimization (dropout removal, Gemm normalization, topological sort)
slimonnx.slim("model.onnx", "model_slim.onnx")

# Enable specific fusions
config = OptimizationConfig(fuse_matmul_add=True, fuse_conv_bn=True, constant_folding=True)
slimonnx.slim("model.onnx", "model_slim.onnx", config=config)
```

## Usage Guide

### Pre-tuned Presets

```python
from slimonnx import SlimONNX, get_preset

config = get_preset("vit_2023")
SlimONNX().slim("vit_model.onnx", "vit_slim.onnx", config=config)
```

Use `all_optimizations(has_batch_dim=True)` to enable all flags.

### Output Validation

```python
from slimonnx import SlimONNX, OptimizationConfig, ValidationConfig

result = SlimONNX().slim(
    "model.onnx",
    "model_slim.onnx",
    config=OptimizationConfig(fuse_conv_bn=True),
    validation=ValidationConfig(validate_outputs=True, num_samples=10),
)
print(result["validation"]["all_match"])
```

### Optimization Flags

All flags default to `False` except `remove_dropout`. Three transforms always run: constant-to-initializer, Gemm normalization, topological reordering.

| Category | Flags |
|----------|-------|
| Conv/BN fusion | `fuse_conv_bn`, `fuse_bn_conv`, `fuse_bn_conv_with_padding`, `fuse_conv_transpose_bn`, `fuse_bn_conv_transpose`, `fuse_depthwise_conv_bn`, `fuse_bn_depthwise_conv` |
| Gemm fusion | `fuse_matmul_add`, `fuse_gemm_gemm`, `fuse_gemm_reshape_bn`, `fuse_bn_reshape_gemm`, `fuse_bn_gemm`, `fuse_transpose_bn_transpose` |
| Simplification | `simplify_conv_to_flatten_gemm`, `remove_redundant_operations`, `constant_folding` |
| Postprocessing | `simplify_node_name`, `remove_dropout` (default: True) |

### API Reference

| Symbol | Description |
|--------|-------------|
| `SlimONNX.slim(onnx_path, target_path, config, validation)` | Optimize and save model |
| `SlimONNX.analyze(onnx_path, config, analysis)` | Analyze structure and detect patterns |
| `SlimONNX.compare(original_path, optimized_path)` | Diff two models by structure and patterns |
| `SlimONNX.validate(onnx_path, config)` | Run ONNX checker, runtime, and graph validation |
| `SlimONNX.detect_patterns(onnx_path, config)` | Find fusible and redundant patterns |
| `SlimONNX.validate_outputs(original_path, optimized_path, validation)` | Compare numerical outputs |
| `SlimONNX.preprocess(onnx_path, target_opset, ...)` | Load, convert opset, and shape-infer a model |
| `get_preset(benchmark_name, model_name)` | Get pre-tuned `OptimizationConfig` for a VNN-COMP benchmark |
| `all_optimizations(has_batch_dim)` | Get `OptimizationConfig` with all flags enabled |

## Project Structure

```
slimonnx/
├── src/slimonnx/
│   ├── slimonnx.py          # SlimONNX class (main entry point)
│   ├── configs.py           # OptimizationConfig, ValidationConfig, AnalysisConfig
│   ├── presets.py           # Pre-tuned configs for VNN-COMP benchmarks
│   ├── utils.py             # Node extraction and shared helpers
│   ├── preprocess/          # Version conversion, cleanup
│   ├── optimize_onnx/       # Optimization passes (fusion, folding, simplification)
│   ├── pattern_detect/      # Pattern detection (fusible, redundant)
│   ├── model_validate/      # ONNX checker, runtime, numerical comparison
│   └── structure_analysis/  # Topology analysis, JSON reports
└── tests/
```

## Tests

```bash
pytest tests/ -v
pytest tests/test_validation.py --preprocess-only
pytest tests/test_benchmarks.py --optimize
pytest tests/test_benchmarks.py --verify
```

## Known Limitations

- Constant folding requires successful shape inference
- BatchNormalization fusion assumes inference mode (`training=False`)
- Models with dynamic shapes have limited optimization opportunities

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
