"""Benchmark execution test for SlimONNX.

Runs optimization on all benchmark models and reports statistics.
Does not compare against baselines, only validates execution success.
"""

__docformat__ = "restructuredtext"
__all__ = ["run_benchmark_optimization", "test_all_benchmarks"]

import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import onnx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slimonnx import SlimONNX
from slim_kwargs import SLIM_KWARGS
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
    load_onnx_model,
)


def run_benchmark_optimization(
    onnx_path: str,
    verbose: bool = False,
    results_dir: str = "results",
) -> dict[str, int | float | str]:
    """Run optimization on a single benchmark model.

    :param onnx_path: Path to ONNX model file
    :param verbose: Whether to print verbose output during optimization
    :param results_dir: Directory for saving optimized models
    :return: Dictionary with results (success, time, node counts, benchmark name)
    """
    benchmark_name = get_benchmark_name(onnx_path)
    opt_config = dict(SLIM_KWARGS[benchmark_name])
    has_batch_dim = if_has_batch_dim(onnx_path)

    # Load original model to get node count
    original_model = load_onnx_model(onnx_path)
    original_node_count = len(original_model.graph.node)

    # Create temp paths
    temp_v22_path = onnx_path.replace(".onnx", "_temp_v22.onnx")
    temp_optimized_path = onnx_path.replace(".onnx", "_temp_optimized.onnx")

    try:
        # Save as v22
        onnx.save(original_model, temp_v22_path)

        # Run optimization
        start_time = time.perf_counter()
        slimonnx = SlimONNX(verbose=False)
        slimonnx.slim(
            temp_v22_path,
            temp_optimized_path,
            has_batch_dim=has_batch_dim,
            **opt_config,
        )
        elapsed_time = time.perf_counter() - start_time

        # Load optimized model to get node count
        optimized_model = onnx.load(temp_optimized_path)
        optimized_node_count = len(optimized_model.graph.node)

        reduction = original_node_count - optimized_node_count
        reduction_pct = (
            (reduction / original_node_count * 100) if original_node_count > 0 else 0
        )

        # Save optimized model to results folder
        # Create results subdirectory: results/benchmark_name/
        result_subdir = Path(results_dir) / benchmark_name
        result_subdir.mkdir(parents=True, exist_ok=True)

        # Save with same filename: results/acasxu_2023/model1.onnx
        model_filename = Path(onnx_path).name
        output_path = result_subdir / model_filename
        shutil.copy(temp_optimized_path, output_path)
        saved_path = str(output_path)

        return {
            "success": True,
            "benchmark": benchmark_name,
            "time": elapsed_time,
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
            "num_optimizations": len(opt_config),
            "error": None,
            "saved_path": saved_path,
        }

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "time": 0.0,
            "original_nodes": original_node_count,
            "optimized_nodes": 0,
            "reduction": 0,
            "reduction_pct": 0.0,
            "num_optimizations": len(opt_config),
            "error": str(e),
            "saved_path": None,
        }

    finally:
        # Clean up temp files
        if os.path.exists(temp_v22_path):
            os.remove(temp_v22_path)
        if os.path.exists(temp_optimized_path):
            os.remove(temp_optimized_path)


def test_all_benchmarks(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    verbose: bool = False,
    results_dir: str = "results",
) -> bool:
    """Run optimization on all benchmark models and report statistics.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark to process
    :param verbose: Whether to print verbose output during optimization
    :param results_dir: Directory for saving optimized models
    :return: True if all tests passed, False otherwise
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    if not onnx_files:
        print(f"No ONNX files found in {benchmark_dir}")
        return False

    print(f"Testing {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks")
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
            "total_reduction": 0,
            "total_original_nodes": 0,
            "total_optimized_nodes": 0,
            "errors": [],
        }
    )

    overall_start = time.perf_counter()

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = run_benchmark_optimization(
            onnx_path,
            verbose=verbose,
            results_dir=results_dir,
        )

        benchmark = result["benchmark"]
        stats = benchmark_stats[benchmark]
        stats["count"] += 1

        if result["success"]:
            stats["success"] += 1
            stats["total_time"] += result["time"]
            stats["total_reduction"] += result["reduction"]
            stats["total_original_nodes"] += result["original_nodes"]
            stats["total_optimized_nodes"] += result["optimized_nodes"]

            saved_info = f" → {result['saved_path']}" if result["saved_path"] else ""
            print(
                f"OK ({result['time']:.2f}s, {result['original_nodes']}->{result['optimized_nodes']} nodes, "
                f"{result['reduction_pct']:.1f}% reduction, {result['num_optimizations']} opts{saved_info})"
            )
        else:
            stats["failed"] += 1
            stats["errors"].append((basename, result["error"]))
            print(f"FAILED: {result['error']}")

    overall_time = time.perf_counter() - overall_start

    # Print summary by benchmark
    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)

    total_success = 0
    total_failed = 0

    for benchmark in sorted(benchmark_stats.keys()):
        stats = benchmark_stats[benchmark]
        avg_time = stats["total_time"] / stats["success"] if stats["success"] > 0 else 0
        avg_reduction_pct = (
            (stats["total_reduction"] / stats["total_original_nodes"] * 100)
            if stats["total_original_nodes"] > 0
            else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Avg time: {avg_time:.2f}s")
        print(
            f"  Total nodes: {stats['total_original_nodes']} -> {stats['total_optimized_nodes']}"
        )
        print(f"  Avg reduction: {avg_reduction_pct:.1f}%")

        if stats["errors"]:
            print(f"  Errors:")
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
    print(f"Total time: {overall_time:.2f}s")
    print(f"Avg time per model: {overall_time/len(onnx_files):.2f}s")

    return total_failed == 0


if __name__ == "__main__":
    success = test_all_benchmarks(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,  # Process all models
        verbose=True,
        results_dir="results",
    )
    sys.exit(0 if success else 1)
