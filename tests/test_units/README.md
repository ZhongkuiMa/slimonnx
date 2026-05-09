# SlimONNX Unit Tests

Unit tests for SlimONNX ONNX graph optimization passes, pattern detection, validation, and utilities.

## Running Tests

```bash
# All unit tests
pytest slimonnx/tests/test_units/ -v

# Single subdirectory
pytest slimonnx/tests/test_units/test_optimize/ -v

# Single file
pytest slimonnx/tests/test_units/test_optimize/test_mm_add.py -v

# Specific test
pytest slimonnx/tests/test_units/test_optimize/test_mm_add.py::TestMatMulAddFusion::test_basic_fusion_success -v

# With coverage
pytest slimonnx/tests/test_units/ --cov=slimonnx --cov-report=term-missing
```

## Test Subdirectories

| Directory | Files | What it tests |
|-----------|-------|---------------|
| `test_optimize/` | 29 | Graph optimization passes (fusion, folding, simplification, removal) |
| `test_pattern_detect/` | 19 | Pattern detection for fuseable op sequences |
| `test_validate/` | 4 | Graph validation, runtime validation, numerical output comparison |
| `test_utils/` | 3 | Model manipulation utility functions |
| `test_api/` | 2 | Public `SlimONNX` class API |
| `test_analyze/` | 2 | Graph structure analysis |
| `test_preprocess/` | 2 | Preprocessing passes (cleanup, opset version conversion) |
| `test_structure_analysis/` | 2 | Topology analysis and structure reporting |

Top-level `test_presets.py` tests optimization preset configurations.

### test_optimize/ highlights

| File | Pass tested |
|------|-------------|
| `test_mm_add.py` | MatMul+Add -> Gemm fusion |
| `test_gemm.py` | Gemm normalization |
| `test_bn_conv.py`, `test_bn_conv_fusion.py` | Conv+BatchNorm fusion |
| `test_bn_gemm.py`, `test_bn_gemm_fusion.py` | Gemm+BatchNorm fusion |
| `test_bn_convtranspose.py` | ConvTranspose+BatchNorm fusion |
| `test_bn_transpose_fusion.py` | Transpose+BatchNorm fusion |
| `test_constant_folding.py` | Constant folding |
| `test_constant_ops.py`, `test_constant_to_initializer.py` | Constant op handling |
| `test_dropout_removal.py` | Dropout removal |
| `test_gemm_gemm_fusion.py` | Gemm chain fusion |
| `test_reshape_optimization.py` | Reshape simplification |
| `test_depthwise_conv.py`, `test_depthwise_conv_fusion.py` | Depthwise conv optimization |
| `test_conv_simplification.py` | Conv simplification |
| `test_gemm_simplification.py` | Gemm simplification |
| `test_redundant.py`, `test_redundant_operations_removal.py` | Redundant op removal |
| `test_core_api.py` | `optimize_onnx()` top-level API |
| `test_error_handling.py` | Error paths and invalid inputs |
| `test_onnx_attrs.py` | ONNX attribute handling |

`*_extended.py` files provide additional coverage variants for the same pass.

## Shared Fixtures (conftest.py)

`conftest.py` provides model-building helpers and test utilities:

- **Builders**: `create_tensor_value_info`, `create_initializer`, `create_minimal_onnx_model`
- **Introspection**: `count_ops_by_type`, `get_nodes_by_type`, `get_node_by_name`, `get_initializer_by_name`, `get_input_names`, `get_initializer_names`
- **Execution**: `run_onnx_model` (onnxruntime, CPU only), `onnx_model_runner` fixture
- **Model factories**: `create_matmul_add_model`, `create_gemm_model`, `create_conv_bn_model`, `create_dropout_model`, `create_convtranspose_bn_model`, `create_depthwise_conv_model`
- **Fixed tensors**: `simple_2x3_matrix`, `simple_3x2_matrix`, `simple_conv_input` (1x3x4x4), `simple_float_model`

## Test Conventions

- Fixed (not random) inputs for reproducibility
- Small tensor shapes (2x3 matrices, 1x3x4x4 conv inputs) for speed
- CPU-only execution via onnxruntime `CPUExecutionProvider`
- Numerical correctness checked with `np.testing.assert_allclose(rtol=1e-5, atol=1e-6)`
- Each test verifies both graph structure (op counts, node types) and numerical output
