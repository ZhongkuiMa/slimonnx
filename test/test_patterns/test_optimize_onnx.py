"""Test optimize_onnx module functions directly.

This module tests individual optimization functions from the optimize_onnx
module using synthetic ONNX models. Tests each optimization in isolation.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "test_matmul_add_fusion",
    "test_conv_bn_fusion",
    "test_dropout_removal",
    "test_redundant_ops_removal",
    "test_constant_folding",
    "test_all_optimizations",
]

import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

# Add parent directories to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from slimonnx.optimize_onnx._optimize import optimize_onnx


def run_onnx_model(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """Run ONNX model with given inputs.

    :param model: ONNX model
    :param inputs: Dictionary of input name -> numpy array
    :return: List of output arrays
    """
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        onnx.save(model, tmp_path)
        session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, inputs)
        return outputs
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compare_outputs(
    original: list[np.ndarray],
    optimized: list[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[bool, float]:
    """Compare outputs from original and optimized models.

    :param original: Outputs from original model
    :param optimized: Outputs from optimized model
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: (match, max_difference)
    """
    if len(original) != len(optimized):
        return False, float("inf")

    max_diff = 0.0
    for orig, opt in zip(original, optimized):
        if orig.shape != opt.shape:
            return False, float("inf")

        diff = np.abs(orig - opt)
        max_diff = max(max_diff, np.max(diff))

        if not np.allclose(orig, opt, rtol=rtol, atol=atol):
            return False, max_diff

    return True, max_diff


def test_matmul_add_fusion() -> bool:
    """Test MatMul + Add to Gemm fusion.

    :return: True if test passed
    """
    print("\nTest: MatMul + Add Fusion")
    print("-" * 80)

    model_path = os.path.join(os.path.dirname(__file__), "models", "matmul_add.onnx")

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 4).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with MatMul+Add fusion
        optimized_model = optimize_onnx(
            original_model,
            fuse_matmul_add=True,
            simplify_gemm=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_conv_bn_fusion() -> bool:
    """Test Conv + BatchNormalization fusion.

    :return: True if test passed
    """
    print("\nTest: Conv + BatchNorm Fusion")
    print("-" * 80)

    model_path = os.path.join(os.path.dirname(__file__), "models", "conv_bn.onnx")

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with Conv+BN fusion
        optimized_model = optimize_onnx(
            original_model,
            fuse_conv_bn=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dropout_removal() -> bool:
    """Test Dropout node removal.

    :return: True if test passed
    """
    print("\nTest: Dropout Removal")
    print("-" * 80)

    model_path = os.path.join(os.path.dirname(__file__), "models", "dropout.onnx")

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 8).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with dropout removal
        optimized_model = optimize_onnx(
            original_model,
            remove_dropout=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_redundant_ops_removal() -> bool:
    """Test redundant operations removal.

    :return: True if test passed
    """
    print("\nTest: Redundant Operations Removal")
    print("-" * 80)

    model_path = os.path.join(os.path.dirname(__file__), "models", "redundant_ops.onnx")

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 8).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with redundant ops removal
        optimized_model = optimize_onnx(
            original_model,
            remove_redundant_operations=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_constant_folding() -> bool:
    """Test constant folding optimization.

    :return: True if test passed
    """
    print("\nTest: Constant Folding")
    print("-" * 80)

    model_path = os.path.join(
        os.path.dirname(__file__), "models", "constant_folding.onnx"
    )

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 4).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with constant folding
        optimized_model = optimize_onnx(
            original_model,
            constant_folding=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_depthwise_conv_bn_fusion() -> bool:
    """Test Depthwise Conv + BatchNormalization fusion.

    :return: True if test passed
    """
    print("\nTest: Depthwise Conv + BatchNorm Fusion")
    print("-" * 80)

    model_path = os.path.join(
        os.path.dirname(__file__), "models", "depthwise_conv_bn.onnx"
    )

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 8, 32, 32).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with depthwise conv+BN fusion
        optimized_model = optimize_onnx(
            original_model,
            fuse_depthwise_conv_bn=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gemm_gemm_fusion() -> bool:
    """Test consecutive Gemm fusion (linear merging).

    :return: True if test passed
    """
    print("\nTest: Gemm + Gemm Fusion (Linear Merging)")
    print("-" * 80)

    model_path = os.path.join(os.path.dirname(__file__), "models", "gemm_gemm.onnx")

    if not os.path.exists(model_path):
        print("  SKIP: Model not found")
        return True

    try:
        original_model = onnx.load(model_path)
        original_nodes = len(original_model.graph.node)

        # Generate input
        inputs = {"input": np.random.randn(1, 8).astype(np.float32)}

        # Run original model
        original_outputs = run_onnx_model(original_model, inputs)

        # Optimize with Gemm+Gemm fusion
        optimized_model = optimize_onnx(
            original_model,
            fuse_gemm_gemm=True,
            simplify_gemm=True,
            has_batch_dim=True,
            verbose=False,
        )

        optimized_nodes = len(optimized_model.graph.node)

        # Run optimized model
        optimized_outputs = run_onnx_model(optimized_model, inputs)

        # Compare
        match, max_diff = compare_outputs(original_outputs, optimized_outputs)

        print(f"  Original nodes: {original_nodes}")
        print(f"  Optimized nodes: {optimized_nodes}")
        print(f"  Node reduction: {original_nodes - optimized_nodes}")
        print(f"  Max output diff: {max_diff:.2e}")
        print(f"  Outputs match: {match}")

        if not match:
            print("  FAILED: Outputs do not match")
            return False

        if optimized_nodes >= original_nodes:
            print("  FAILED: No node reduction")
            return False

        print("  PASSED")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_all_optimizations() -> bool:
    """Run all optimization tests.

    :return: True if all tests passed
    """
    print("=" * 80)
    print("Testing optimize_onnx module functions")
    print("=" * 80)

    tests = [
        ("MatMul+Add Fusion", test_matmul_add_fusion),
        ("Conv+BN Fusion", test_conv_bn_fusion),
        ("Dropout Removal", test_dropout_removal),
        ("Redundant Ops Removal", test_redundant_ops_removal),
        ("Constant Folding", test_constant_folding),
        ("Depthwise Conv+BN Fusion", test_depthwise_conv_bn_fusion),
        ("Gemm+Gemm Fusion", test_gemm_gemm_fusion),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\nTest '{test_name}' crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, p in results:
            if not p:
                print(f"  - {name}")

    success = failed == 0
    if success:
        print("\nSUCCESS: All optimization tests passed")
    else:
        print("\nFAILURE: Some optimization tests failed")

    return success


if __name__ == "__main__":
    success = test_all_optimizations()
    sys.exit(0 if success else 1)
