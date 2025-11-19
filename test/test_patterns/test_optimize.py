"""Test optimization on synthetic ONNX models.

This module tests SlimONNX optimization functionality using
small synthetic models with known patterns. Tests both structural
changes (node count reduction) and numerical correctness.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_single_optimization", "test_all_optimizations"]

import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from slimonnx import SlimONNX, OptimizationConfig


def run_onnx_model(model_path: str, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run ONNX model with given inputs.

    :param model_path: Path to ONNX model
    :param inputs: Dictionary of input name -> numpy array
    :return: List of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    return outputs


def compare_outputs(
    original_outputs: list[np.ndarray],
    optimized_outputs: list[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Compare outputs from original and optimized models.

    :param original_outputs: Outputs from original model
    :param optimized_outputs: Outputs from optimized model
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: (match, max_difference)
    """
    if len(original_outputs) != len(optimized_outputs):
        return False, float("inf")

    max_diff = 0.0
    for orig, opt in zip(original_outputs, optimized_outputs):
        if orig.shape != opt.shape:
            return False, float("inf")

        diff = np.abs(orig - opt)
        max_diff = max(max_diff, np.max(diff))

        if not np.allclose(orig, opt, rtol=rtol, atol=atol):
            return False, max_diff

    return True, max_diff


def test_single_optimization(
    onnx_path: str,
    optimization_config: OptimizationConfig,
    expected_node_reduction: int | None = None,
) -> bool:
    """Test optimization on a single model.

    :param onnx_path: Path to ONNX model
    :param optimization_config: Optimization configuration
    :param expected_node_reduction: Expected node count reduction (optional)
    :return: True if optimization succeeded and is correct
    """
    slimonnx = SlimONNX(verbose=False)

    try:
        # Load original model
        original_model = onnx.load(onnx_path)
        original_node_count = len(original_model.graph.node)

        # Get input info
        input_info = {}
        for inp in original_model.graph.input:
            # Skip if input is an initializer
            initializer_names = [init.name for init in original_model.graph.initializer]
            if inp.name in initializer_names:
                continue

            shape = [
                dim.dim_value if dim.dim_value > 0 else 1
                for dim in inp.type.tensor_type.shape.dim
            ]
            input_info[inp.name] = shape

        # Generate random inputs
        inputs = {
            name: np.random.randn(*shape).astype(np.float32)
            for name, shape in input_info.items()
        }

        # Run original model
        original_outputs = run_onnx_model(onnx_path, inputs)

        # Optimize model
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            optimized_path = tmp.name

        slimonnx.slim(onnx_path, optimized_path, config=optimization_config)

        # Load optimized model
        optimized_model = onnx.load(optimized_path)
        optimized_node_count = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_path, inputs)

        # Compare outputs
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        # Check structural change
        node_reduction = original_node_count - optimized_node_count

        # Cleanup
        os.unlink(optimized_path)

        # Report results
        print(f"  Original nodes: {original_node_count}")
        print(f"  Optimized nodes: {optimized_node_count}")
        print(f"  Node reduction: {node_reduction}")
        print(f"  Max output difference: {max_diff:.2e}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if (
            expected_node_reduction is not None
            and node_reduction < expected_node_reduction
        ):
            print(
                f"  FAILED: Expected at least {expected_node_reduction} node reduction, got {node_reduction}"
            )
            return False

        print("  OK: Optimization successful and correct")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_optimizations() -> bool:
    """Test optimization on all synthetic models.

    :return: True if all tests passed
    """
    # Pattern -> (config, expected_node_reduction)
    test_cases = {
        "matmul_add.onnx": (
            OptimizationConfig(fuse_matmul_add=True, has_batch_dim=True),
            1,
        ),
        "conv_bn.onnx": (
            OptimizationConfig(fuse_conv_bn=True, has_batch_dim=True),
            1,
        ),
        "bn_conv.onnx": (
            OptimizationConfig(fuse_bn_conv=True, has_batch_dim=True),
            1,
        ),
        "convtranspose_bn.onnx": (
            OptimizationConfig(fuse_convtransposed_bn=True, has_batch_dim=True),
            1,
        ),
        "bn_convtranspose.onnx": (
            OptimizationConfig(fuse_bn_convtransposed=True, has_batch_dim=True),
            1,
        ),
        "depthwise_conv_bn.onnx": (
            OptimizationConfig(fuse_depthwise_conv_bn=True, has_batch_dim=True),
            1,
        ),
        "gemm_reshape_bn.onnx": (
            OptimizationConfig(fuse_gemm_reshape_bn=True, has_batch_dim=True),
            2,
        ),
        "bn_reshape_gemm.onnx": (
            OptimizationConfig(fuse_bn_reshape_gemm=True, has_batch_dim=True),
            2,
        ),
        "bn_gemm.onnx": (
            OptimizationConfig(fuse_bn_gemm=True, has_batch_dim=True),
            1,
        ),
        "transpose_bn_transpose.onnx": (
            OptimizationConfig(fuse_transpose_bn_transpose=True, has_batch_dim=True),
            2,
        ),
        "gemm_gemm.onnx": (
            OptimizationConfig(fuse_gemm_gemm=True, has_batch_dim=True),
            1,
        ),
        "dropout.onnx": (
            OptimizationConfig(remove_dropout=True, has_batch_dim=True),
            1,
        ),
        "redundant_ops.onnx": (
            OptimizationConfig(remove_redundant_operations=True, has_batch_dim=True),
            4,
        ),
        "constant_folding.onnx": (
            OptimizationConfig(constant_folding=True, has_batch_dim=True),
            2,
        ),
    }

    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        print("Run create_patterns.py first to generate test models")
        return False

    print("Testing optimization on synthetic models")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for model_file, (config, expected_reduction) in test_cases.items():
        model_path = os.path.join(models_dir, model_file)

        if not os.path.exists(model_path):
            print(f"SKIP: {model_file} (file not found)")
            continue

        print(f"\nTesting: {model_file}")
        print("-" * 80)

        total_tests += 1
        if test_single_optimization(model_path, config, expected_reduction):
            passed_tests += 1
        else:
            failed_tests.append(model_file)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")

    success = len(failed_tests) == 0
    if success:
        print("\nSUCCESS: All optimization tests passed")
    else:
        print("\nFAILURE: Some optimization tests failed")

    return success


if __name__ == "__main__":
    success = test_all_optimizations()
    sys.exit(0 if success else 1)
