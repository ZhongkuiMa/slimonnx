"""Test preprocessing and model validation for SlimONNX.

Combines two types of testing:
1. Preprocessing - Model loading, version conversion, shape inference
2. Model Validation - ONNX checker, runtime validator, graph validation
"""

__docformat__ = "restructuredtext"
__all__ = [
    "test_preprocess",
    "test_all_preprocess",
    "test_validation",
    "test_all_validation",
    "main",
]

import os
from collections import defaultdict

from slimonnx import SlimONNX, OptimizationConfig
from slimonnx.test.utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
)


def test_preprocess(onnx_path: str, target_opset: int | None = None) -> dict:
    """Test preprocessing on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :param target_opset: Target opset version (None = keep original)
    :return: Preprocessing test result dictionary
    """
    slimonnx = SlimONNX()
    benchmark_name = get_benchmark_name(onnx_path)

    try:
        model = slimonnx.preprocess(
            onnx_path,
            target_opset=target_opset,
            infer_shapes=True,
        )

        original_opset = model.opset_import[0].version if model.opset_import else 0
        ir_version = model.ir_version
        node_count = len(model.graph.node)
        input_count = len(model.graph.input)
        output_count = len(model.graph.output)
        initializer_count = len(model.graph.initializer)

        has_shapes = all(
            hasattr(inp, "type")
            and hasattr(inp.type, "tensor_type")
            and hasattr(inp.type.tensor_type, "shape")
            for inp in model.graph.input
        )

        return {
            "success": True,
            "benchmark": benchmark_name,
            "opset": original_opset,
            "ir_version": ir_version,
            "node_count": node_count,
            "input_count": input_count,
            "output_count": output_count,
            "initializer_count": initializer_count,
            "has_shapes": has_shapes,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "opset": 0,
            "ir_version": 0,
            "node_count": 0,
            "input_count": 0,
            "output_count": 0,
            "initializer_count": 0,
            "has_shapes": False,
            "error": str(e),
        }


def test_all_preprocess(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    target_opset: int | None = None,
) -> dict:
    """Test preprocessing on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark
    :param target_opset: Target opset version (None = keep original)
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    print(f"Testing preprocessing on {len(onnx_files)} models")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    opset_counts = defaultdict(int)
    models_with_shapes = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = test_preprocess(onnx_path, target_opset)

        if result["success"]:
            success_count += 1
            opset_counts[result["opset"]] += 1
            if result["has_shapes"]:
                models_with_shapes += 1

            print(
                f"OK (opset {result['opset']}, IR {result['ir_version']}, {result['node_count']} nodes, shapes: {result['has_shapes']})"
            )
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()
    print(
        f"Models with shape inference: {models_with_shapes}/{success_count} ({models_with_shapes/success_count*100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print("\nOpset distribution:")
    for opset in sorted(opset_counts.keys()):
        print(f"  Opset {opset}: {opset_counts[opset]}")

    return {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "models_with_shapes": models_with_shapes,
        "opset_counts": dict(opset_counts),
    }


def test_validation(onnx_path: str) -> dict:
    """Test validation on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :return: Validation test result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    config = OptimizationConfig(has_batch_dim=has_batch_dim)
    slimonnx = SlimONNX()

    try:
        validation = slimonnx.validate(onnx_path, config=config)

        return {
            "success": True,
            "benchmark": benchmark_name,
            "is_valid": validation["is_valid"],
            "onnx_checker_valid": validation["onnx_checker"]["valid"],
            "runtime_can_load": validation["runtime"]["can_load"],
            "dead_nodes": len(validation["dead_nodes"]),
            "broken_connections": len(validation["broken_connections"]),
            "orphan_initializers": len(validation["orphan_initializers"]),
            "type_errors": len(validation["type_errors"]),
            "shape_errors": len(validation["shape_errors"]),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "is_valid": False,
            "onnx_checker_valid": False,
            "runtime_can_load": False,
            "dead_nodes": 0,
            "broken_connections": 0,
            "orphan_initializers": 0,
            "type_errors": 0,
            "shape_errors": 0,
            "error": str(e),
        }


def test_all_validation(
    benchmark_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> dict:
    """Test validation on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    print(f"Testing validation on {len(onnx_files)} models")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    valid_count = 0
    checker_valid_count = 0
    runtime_loadable_count = 0
    total_dead_nodes = 0
    total_issues = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = test_validation(onnx_path)

        if result["success"]:
            success_count += 1
            if result["is_valid"]:
                valid_count += 1
            if result["onnx_checker_valid"]:
                checker_valid_count += 1
            if result["runtime_can_load"]:
                runtime_loadable_count += 1

            total_dead_nodes += result["dead_nodes"]
            issues = (
                result["dead_nodes"]
                + result["broken_connections"]
                + result["orphan_initializers"]
                + result["type_errors"]
                + result["shape_errors"]
            )
            total_issues += issues

            status = "VALID" if result["is_valid"] else "INVALID"
            print(f"{status} (issues: {issues})")
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()
    print(
        f"Valid models: {valid_count}/{success_count} ({valid_count/success_count*100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print(
        f"ONNX checker valid: {checker_valid_count}/{success_count} ({checker_valid_count/success_count*100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print(
        f"Runtime loadable: {runtime_loadable_count}/{success_count} ({runtime_loadable_count/success_count*100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print()
    print(f"Total dead nodes: {total_dead_nodes}")
    print(f"Total issues: {total_issues}")

    return {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "valid_count": valid_count,
        "checker_valid_count": checker_valid_count,
        "runtime_loadable_count": runtime_loadable_count,
        "total_dead_nodes": total_dead_nodes,
        "total_issues": total_issues,
    }


def main() -> None:
    """Main entry point for script execution."""
    import sys

    if "--preprocess-only" in sys.argv:
        opset = None
        if "--opset" in sys.argv:
            idx = sys.argv.index("--opset")
            if idx + 1 < len(sys.argv):
                opset = int(sys.argv[idx + 1])
        test_all_preprocess(target_opset=opset)
    elif "--validate-only" in sys.argv:
        test_all_validation()
    else:
        print("Running both preprocessing and validation tests...\n")
        test_all_preprocess()
        print("\n")
        test_all_validation()


if __name__ == "__main__":
    main()
