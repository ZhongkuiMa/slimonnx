"""Test pattern detection on synthetic ONNX models.

This module tests SlimONNX pattern detection functionality using
small synthetic models with known patterns.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_single_pattern", "test_all_patterns"]

import os
import sys

from slimonnx import SlimONNX, OptimizationConfig


def test_single_pattern(onnx_path: str, expected_pattern: str) -> bool:
    """Test pattern detection on a single model.

    :param onnx_path: Path to ONNX model
    :param expected_pattern: Expected pattern name to detect
    :return: True if pattern detected correctly
    """
    config = OptimizationConfig(has_batch_dim=True)
    slimonnx = SlimONNX()

    try:
        patterns = slimonnx.detect_patterns(onnx_path, config=config)

        # Check if expected pattern was detected
        if expected_pattern not in patterns:
            print(f"FAILED: Pattern '{expected_pattern}' not in results")
            return False

        pattern_data = patterns[expected_pattern]
        detected_count = pattern_data["count"]

        if detected_count == 0:
            print(f"FAILED: Pattern '{expected_pattern}' detected but count is 0")
            return False

        print(f"OK: Detected {detected_count} instance(s) of '{expected_pattern}'")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_all_patterns() -> bool:
    """Test pattern detection on all synthetic models.

    :return: True if all tests passed
    """
    # Pattern name -> expected pattern to detect
    test_cases = {
        "matmul_add.onnx": "matmul_add",
        "conv_bn.onnx": "conv_bn",
        "bn_conv.onnx": "bn_conv",
        "convtranspose_bn.onnx": "convtranspose_bn",
        "bn_convtranspose.onnx": "bn_convtranspose",
        "depthwise_conv_bn.onnx": "depthwise_conv_bn",
        "gemm_reshape_bn.onnx": "gemm_reshape_bn",
        "bn_reshape_gemm.onnx": "bn_reshape_gemm",
        "bn_gemm.onnx": "bn_gemm",
        "transpose_bn_transpose.onnx": "transpose_bn_transpose",
        "gemm_gemm.onnx": "gemm_gemm",
        "dropout.onnx": "dropout",
        "redundant_ops.onnx": ["add_zero", "sub_zero", "mul_one", "div_one"],
        "constant_folding.onnx": "constant_foldable",
    }

    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory not found: {models_dir}")
        print("Run create_patterns.py first to generate test models")
        return False

    print("Testing pattern detection on synthetic models")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for model_file, expected in test_cases.items():
        model_path = os.path.join(models_dir, model_file)

        if not os.path.exists(model_path):
            print(f"SKIP: {model_file} (file not found)")
            continue

        print(f"\nTesting: {model_file}")
        print("-" * 80)

        # Handle multiple expected patterns
        if isinstance(expected, list):
            test_passed = True
            for pattern in expected:
                total_tests += 1
                if test_single_pattern(model_path, pattern):
                    passed_tests += 1
                else:
                    test_passed = False
                    failed_tests.append(f"{model_file} ({pattern})")
        else:
            total_tests += 1
            if test_single_pattern(model_path, expected):
                passed_tests += 1
            else:
                failed_tests.append(f"{model_file} ({expected})")

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
        print("\nSUCCESS: All pattern detection tests passed")
    else:
        print("\nFAILURE: Some pattern detection tests failed")

    return success


if __name__ == "__main__":
    success = test_all_patterns()
    sys.exit(0 if success else 1)
