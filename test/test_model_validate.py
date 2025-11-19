"""Test model validation functionality for SlimONNX.

Tests ONNX checker, runtime validator, and graph validation.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_model_validate", "test_all_model_validate"]

import os
import sys
from collections import defaultdict
from pathlib import Path

# Add parent and rover_alpha directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from slimonnx import SlimONNX, OptimizationConfig
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
)


def test_one_model_validate(
    onnx_path: str,
    verbose: bool = False,
) -> dict:
    """Test validation on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :param verbose: Print detailed output
    :return: Validation test result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    config = OptimizationConfig(has_batch_dim=has_batch_dim)
    slimonnx = SlimONNX(verbose=verbose)

    try:
        # Validate model
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


def test_all_model_validate(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    verbose: bool = False,
    results_dir: str = "results/model_validate",
) -> bool:
    """Test validation on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark to process
    :param verbose: Print detailed output
    :param results_dir: Directory to save test results
    :return: True if all tests passed, False otherwise
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    if not onnx_files:
        print(f"No ONNX files found in {benchmark_dir}")
        return False

    print(
        f"Testing model validation on {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks"
    )
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "valid_models": 0,
            "onnx_checker_pass": 0,
            "runtime_loadable": 0,
            "total_issues": 0,
            "errors": [],
        }
    )

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_one_model_validate(onnx_path, verbose=verbose)

        benchmark = result["benchmark"]
        stats = benchmark_stats[benchmark]
        stats["count"] += 1

        if result["success"]:
            stats["success"] += 1
            if result["is_valid"]:
                stats["valid_models"] += 1
            if result["onnx_checker_valid"]:
                stats["onnx_checker_pass"] += 1
            if result["runtime_can_load"]:
                stats["runtime_loadable"] += 1

            issues = (
                result["dead_nodes"]
                + result["broken_connections"]
                + result["orphan_initializers"]
                + result["type_errors"]
                + result["shape_errors"]
            )
            stats["total_issues"] += issues

            status = "VALID" if result["is_valid"] else "INVALID"
            issues_str = f"{issues} issues" if issues > 0 else "no issues"
            print(
                f"OK ({status}, checker={'pass' if result['onnx_checker_valid'] else 'fail'}, "
                f"runtime={'ok' if result['runtime_can_load'] else 'fail'}, {issues_str})"
            )
        else:
            stats["failed"] += 1
            stats["errors"].append((basename, result["error"]))
            print(f"FAILED: {result['error']}")

    # Print summary by benchmark
    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    total_success = 0
    total_failed = 0

    for benchmark in sorted(benchmark_stats.keys()):
        stats = benchmark_stats[benchmark]
        valid_pct = (
            (stats["valid_models"] / stats["success"] * 100)
            if stats["success"] > 0
            else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(
            f"  Valid models: {stats['valid_models']}/{stats['success']} ({valid_pct:.1f}%)"
        )
        print(f"  ONNX checker pass: {stats['onnx_checker_pass']}/{stats['success']}")
        print(f"  Runtime loadable: {stats['runtime_loadable']}/{stats['success']}")
        print(f"  Total issues: {stats['total_issues']}")

        if stats["errors"]:
            print("  Errors:")
            for basename, error in stats["errors"]:
                print(f"    {basename}: {error}")

        total_success += stats["success"]
        total_failed += stats["failed"]

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {total_success/len(onnx_files)*100:.1f}%")

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "test_results.txt", "w") as f:
        f.write(f"Total models: {len(onnx_files)}\n")
        f.write(f"Success: {total_success}\n")
        f.write(f"Failed: {total_failed}\n")
        f.write(f"Success rate: {total_success/len(onnx_files)*100:.1f}%\n")

    return total_failed == 0


if __name__ == "__main__":
    success = test_all_model_validate(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,
        verbose=False,
    )
    sys.exit(0 if success else 1)
