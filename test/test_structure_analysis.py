"""Test structure analysis functionality for SlimONNX.

Tests model structure analysis, topology extraction, and reporting.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_structure_analysis", "test_all_structure_analysis"]

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


def test_one_structure_analysis(
    onnx_path: str,
    verbose: bool = False,
) -> dict:
    """Test structure analysis on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :param verbose: Print detailed output
    :return: Structure analysis test result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    config = OptimizationConfig(has_batch_dim=has_batch_dim)
    slimonnx = SlimONNX(verbose=verbose)

    try:
        # Analyze model (includes structure analysis)
        analysis = slimonnx.analyze(onnx_path, config=config)

        structure = analysis["structure"]

        return {
            "success": True,
            "benchmark": benchmark_name,
            "node_count": structure["node_count"],
            "initializer_count": structure["initializer_count"],
            "num_inputs": structure["num_inputs"],
            "num_outputs": structure["num_outputs"],
            "has_shapes": structure["has_shapes"],
            "op_types": len(structure["op_type_counts"]),
            "top_ops": dict(
                sorted(
                    structure["op_type_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "node_count": 0,
            "initializer_count": 0,
            "num_inputs": 0,
            "num_outputs": 0,
            "has_shapes": False,
            "op_types": 0,
            "top_ops": {},
            "error": str(e),
        }


def test_all_structure_analysis(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    verbose: bool = False,
    results_dir: str = "results/structure_analysis",
) -> bool:
    """Test structure analysis on all benchmark models.

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
        f"Testing structure analysis on {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks"
    )
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_nodes": 0,
            "total_initializers": 0,
            "has_shapes_count": 0,
            "errors": [],
        }
    )

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_one_structure_analysis(onnx_path, verbose=verbose)

        benchmark = result["benchmark"]
        stats = benchmark_stats[benchmark]
        stats["count"] += 1

        if result["success"]:
            stats["success"] += 1
            stats["total_nodes"] += result["node_count"]
            stats["total_initializers"] += result["initializer_count"]
            if result["has_shapes"]:
                stats["has_shapes_count"] += 1

            top_ops_str = ", ".join(
                f"{op}={count}" for op, count in list(result["top_ops"].items())[:3]
            )
            print(
                f"OK ({result['node_count']} nodes, {result['op_types']} op types, top: {top_ops_str})"
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
        avg_nodes = (
            stats["total_nodes"] / stats["success"] if stats["success"] > 0 else 0
        )
        avg_inits = (
            stats["total_initializers"] / stats["success"]
            if stats["success"] > 0
            else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Avg nodes: {avg_nodes:.0f}")
        print(f"  Avg initializers: {avg_inits:.0f}")
        print(f"  Has shapes: {stats['has_shapes_count']}/{stats['success']}")

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
    success = test_all_structure_analysis(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,
        verbose=False,
    )
    sys.exit(0 if success else 1)
