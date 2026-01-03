"""Test preprocessing and model validation for SlimONNX.

Combines two types of testing:
1. Preprocessing - Model loading, version conversion, shape inference
2. Model Validation - ONNX checker, runtime validator, graph validation
"""

__docformat__ = "restructuredtext"
__all__ = [
    "main",
    "run_all_preprocess",
    "run_all_validation",
    "run_preprocess_test",
    "run_validation_test",
    "test_preprocess_benchmarks",
    "test_validation_benchmarks",
]

from collections import defaultdict
from pathlib import Path

from slimonnx import OptimizationConfig
from slimonnx.slimonnx import SlimONNX
from tests.test_benchmarks.benchmark_utils import (
    find_benchmark_folders,
    find_onnx_files_from_instances,
    get_benchmark_name,
)
from tests.utils import if_has_batch_dim


def run_preprocess_test(onnx_path: str, target_opset: int | None = None) -> dict:
    """Run preprocessing test on a single ONNX model.

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

    except (OSError, ValueError, AttributeError) as error:
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
            "error": str(error),
        }


def run_all_preprocess(
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
    benchmark_dirs = find_benchmark_folders(benchmark_dir)
    onnx_files = find_onnx_files_from_instances(benchmark_dirs, num_limit=max_per_benchmark)

    print(f"Testing preprocessing on {len(onnx_files)} models")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    opset_counts: defaultdict[int, int] = defaultdict(int)
    models_with_shapes = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = Path(onnx_path).name
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = run_preprocess_test(onnx_path, target_opset)

        if result["success"]:
            success_count += 1
            opset_counts[result["opset"]] += 1
            if result["has_shapes"]:
                models_with_shapes += 1

            status_msg = (
                f"OK (opset {result['opset']}, IR {result['ir_version']}, "
                f"{result['node_count']} nodes, shapes: {result['has_shapes']})"
            )
            print(status_msg)
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
    shape_msg = (
        f"Models with shape inference: {models_with_shapes}/{success_count} "
        f"({models_with_shapes / success_count * 100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print(shape_msg)
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


def run_validation_test(onnx_path: str) -> dict:
    """Run validation test on a single ONNX model.

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

    except (OSError, ValueError, AttributeError, RuntimeError) as error:
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
            "error": str(error),
        }


def run_all_validation(benchmark_dir: str = "benchmarks", max_per_benchmark: int = 20) -> dict:
    """Test validation on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmark_folders(benchmark_dir)
    onnx_files = find_onnx_files_from_instances(benchmark_dirs, num_limit=max_per_benchmark)

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
        basename = Path(onnx_path).name
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = run_validation_test(onnx_path)

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
        f"Valid models: {valid_count}/{success_count} ({valid_count / success_count * 100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    checker_msg = (
        f"ONNX checker valid: {checker_valid_count}/{success_count} "
        f"({checker_valid_count / success_count * 100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print(checker_msg)
    runtime_msg = (
        f"Runtime loadable: {runtime_loadable_count}/{success_count} "
        f"({runtime_loadable_count / success_count * 100:.1f}%)"
        if success_count > 0
        else "N/A"
    )
    print(runtime_msg)
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


def test_preprocess_benchmarks() -> None:
    """Pytest: Test preprocessing on all benchmark models."""
    from pathlib import Path

    import pytest

    benchmark_dir = Path(__file__).parent / "vnncomp2024_benchmarks"
    if not benchmark_dir.exists():
        pytest.skip(f"Benchmark directory not found: {benchmark_dir}")

    result = run_all_preprocess(str(benchmark_dir))
    assert result["success"] > 0, "No models successfully preprocessed"
    assert result["failed"] == 0, f"Preprocessing failed for {result['failed']} models"


def test_validation_benchmarks() -> None:
    """Pytest: Test validation on all benchmark models."""
    from pathlib import Path

    import pytest

    benchmark_dir = Path(__file__).parent / "vnncomp2024_benchmarks"
    if not benchmark_dir.exists():
        pytest.skip(f"Benchmark directory not found: {benchmark_dir}")

    result = run_all_validation(str(benchmark_dir))
    assert result["success"] > 0, "No models successfully validated"
    assert result["failed"] == 0, f"Validation failed for {result['failed']} models"
    assert result["valid_count"] == result["success"], (
        f"Some models are invalid: {result['success'] - result['valid_count']} models"
    )


def main() -> None:
    """Run the validation script."""
    import sys

    if "--preprocess-only" in sys.argv:
        opset = None
        if "--opset" in sys.argv:
            idx = sys.argv.index("--opset")
            if idx + 1 < len(sys.argv):
                opset = int(sys.argv[idx + 1])
        run_all_preprocess(target_opset=opset)
    elif "--validate-only" in sys.argv:
        run_all_validation()
    else:
        print("Running both preprocessing and validation tests...\n")
        run_all_preprocess()
        print("\n")
        run_all_validation()


if __name__ == "__main__":
    main()
