"""Optimization execution test for SlimONNX.

Runs optimization on all benchmark models and reports statistics.
Does not compare against baselines, only validates execution success.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_optimize", "test_all_optimize"]

import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import onnx

from slimonnx import SlimONNX, get_preset
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    load_onnx_model,
)


def calculate_model_parameters(model: onnx.ModelProto) -> tuple[int, float]:
    """Calculate total parameters and memory usage of an ONNX model.

    :param model: ONNX model
    :return: Tuple of (total_parameters, memory_mb)
    """
    total_params = 0
    total_bytes = 0

    for tensor in model.graph.initializer:
        size = 1
        for dim in tensor.dims:
            size *= dim

        total_params += size

        # Estimate bytes based on data type
        if tensor.data_type == 1:  # FLOAT
            total_bytes += size * 4
        elif tensor.data_type == 11:  # DOUBLE
            total_bytes += size * 8
        elif tensor.data_type in {6, 7}:  # INT32, INT64
            total_bytes += size * 4
        else:
            total_bytes += size * 4

    memory_mb = total_bytes / (1024 * 1024)
    return total_params, memory_mb


def test_one_optimize(
    onnx_path: str,
    verbose: bool = False,
    results_dir: str = "results/optimize_onnx",
) -> dict[str, int | float | str]:
    """Run optimization on a single benchmark model.

    :param onnx_path: Path to ONNX model file
    :param verbose: Whether to print verbose output during optimization
    :param results_dir: Directory for saving optimized models
    :return: Dictionary with results (success, time, node counts, benchmark name)
    """
    benchmark_name = get_benchmark_name(onnx_path)
    config = get_preset(benchmark_name)

    # Load original model to get metrics
    original_model = load_onnx_model(onnx_path)
    original_node_count = len(original_model.graph.node)
    original_params, original_memory = calculate_model_parameters(original_model)

    # Create temp paths
    temp_v22_path = onnx_path.replace(".onnx", "_temp_v22.onnx")
    temp_optimized_path = onnx_path.replace(".onnx", "_temp_optimized.onnx")

    try:
        # Save as v22
        onnx.save(original_model, temp_v22_path)

        # Detect patterns before optimization
        slimonnx = SlimONNX(verbose=verbose)
        patterns_before = slimonnx.detect_patterns(temp_v22_path, config=config)
        total_patterns = sum(p["count"] for p in patterns_before.values())

        # Run optimization
        start_time = time.perf_counter()
        slimonnx.slim(
            temp_v22_path,
            temp_optimized_path,
            config=config,
        )
        elapsed_time = time.perf_counter() - start_time

        # Load optimized model to get metrics
        optimized_model = onnx.load(temp_optimized_path)
        optimized_node_count = len(optimized_model.graph.node)
        optimized_params, optimized_memory = calculate_model_parameters(optimized_model)

        node_reduction = original_node_count - optimized_node_count
        node_reduction_pct = (
            (node_reduction / original_node_count * 100)
            if original_node_count > 0
            else 0
        )

        param_reduction = original_params - optimized_params
        memory_reduction = original_memory - optimized_memory

        # Save optimized model to results folder
        result_subdir = Path(results_dir) / benchmark_name
        result_subdir.mkdir(parents=True, exist_ok=True)

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
            "node_reduction": node_reduction,
            "node_reduction_pct": node_reduction_pct,
            "original_params": original_params,
            "optimized_params": optimized_params,
            "param_reduction": param_reduction,
            "original_memory": original_memory,
            "optimized_memory": optimized_memory,
            "memory_reduction": memory_reduction,
            "patterns_detected": total_patterns,
            "patterns_detail": patterns_before,
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
            "node_reduction": 0,
            "node_reduction_pct": 0.0,
            "original_params": original_params,
            "optimized_params": 0,
            "param_reduction": 0,
            "original_memory": original_memory,
            "optimized_memory": 0.0,
            "memory_reduction": 0.0,
            "patterns_detected": 0,
            "patterns_detail": {},
            "error": str(e),
            "saved_path": None,
        }

    finally:
        # Clean up temp files
        if os.path.exists(temp_v22_path):
            os.remove(temp_v22_path)
        if os.path.exists(temp_optimized_path):
            os.remove(temp_optimized_path)


def test_all_optimize(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    verbose: bool = False,
    results_dir: str = "results/optimize_onnx",
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

    print(
        f"Testing optimization on {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks"
    )
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
            "total_node_reduction": 0,
            "total_original_nodes": 0,
            "total_optimized_nodes": 0,
            "total_param_reduction": 0,
            "total_original_params": 0,
            "total_optimized_params": 0,
            "total_memory_reduction": 0.0,
            "total_original_memory": 0.0,
            "total_optimized_memory": 0.0,
            "total_patterns": 0,
            "pattern_counts": defaultdict(int),
            "errors": [],
        }
    )

    overall_start = time.perf_counter()

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_one_optimize(
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
            stats["total_node_reduction"] += result["node_reduction"]
            stats["total_original_nodes"] += result["original_nodes"]
            stats["total_optimized_nodes"] += result["optimized_nodes"]
            stats["total_param_reduction"] += result["param_reduction"]
            stats["total_original_params"] += result["original_params"]
            stats["total_optimized_params"] += result["optimized_params"]
            stats["total_memory_reduction"] += result["memory_reduction"]
            stats["total_original_memory"] += result["original_memory"]
            stats["total_optimized_memory"] += result["optimized_memory"]
            stats["total_patterns"] += result["patterns_detected"]

            # Track individual pattern counts
            for pattern_name, pattern_info in result["patterns_detail"].items():
                stats["pattern_counts"][pattern_name] += pattern_info["count"]

            print(
                f"OK ({result['time']:.2f}s, {result['original_nodes']}->{result['optimized_nodes']} nodes, "
                f"{result['node_reduction_pct']:.1f}% reduction, {result['patterns_detected']} patterns, "
                f"{result['original_memory']:.1f}->{result['optimized_memory']:.1f}MB)"
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
        success_count = stats["success"]

        avg_time = stats["total_time"] / success_count if success_count > 0 else 0
        avg_node_reduction_pct = (
            (stats["total_node_reduction"] / stats["total_original_nodes"] * 100)
            if stats["total_original_nodes"] > 0
            else 0
        )
        avg_param_reduction_pct = (
            (stats["total_param_reduction"] / stats["total_original_params"] * 100)
            if stats["total_original_params"] > 0
            else 0
        )
        avg_memory_reduction = (
            stats["total_memory_reduction"] / success_count if success_count > 0 else 0
        )
        avg_patterns = (
            stats["total_patterns"] / success_count if success_count > 0 else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Avg time: {avg_time:.2f}s")
        print(
            f"  Nodes: {stats['total_original_nodes']} -> {stats['total_optimized_nodes']} "
            f"({avg_node_reduction_pct:.1f}% reduction)"
        )
        print(
            f"  Params: {stats['total_original_params']:,} -> {stats['total_optimized_params']:,} "
            f"({avg_param_reduction_pct:.1f}% reduction)"
        )
        print(
            f"  Memory: {stats['total_original_memory']:.1f} -> {stats['total_optimized_memory']:.1f} MB "
            f"({avg_memory_reduction:.1f} MB avg reduction)"
        )
        print(f"  Avg patterns detected: {avg_patterns:.1f}")

        # Show top patterns if any
        if stats["pattern_counts"]:
            top_patterns = sorted(
                stats["pattern_counts"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            pattern_str = ", ".join(f"{p}={c}" for p, c in top_patterns)
            print(f"  Top patterns: {pattern_str}")

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
    print(f"Total time: {overall_time:.2f}s")
    print(f"Avg time per model: {overall_time/len(onnx_files):.2f}s")

    return total_failed == 0


if __name__ == "__main__":
    success = test_all_optimize(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,
        verbose=False,
        results_dir="results/optimize_onnx",
    )
    sys.exit(0 if success else 1)
