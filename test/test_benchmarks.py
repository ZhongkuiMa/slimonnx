"""Run optimization tests on all benchmarks without baseline comparison.

This module tests SlimONNX optimization on all benchmark models:
- Saves optimized models to results/test_benchmarks/
- Prints comprehensive statistics (node reduction, time, memory, patterns)
- Does not compare against baselines (use test_baselines.py for that)
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_benchmark", "test_all_benchmarks"]

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import onnx

from slimonnx import SlimONNX, get_preset
from slimonnx.slimonnx.analyze_structure import analyze_model
from slimonnx.test.utils import (
    find_onnx_files_from_instances,
    find_benchmark_folders,
    get_benchmark_name,
    load_onnx_model,
)


def _calculate_model_parameters(model: onnx.ModelProto) -> tuple[int, float]:
    """Calculate total parameters and memory usage.

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

        if tensor.data_type == 1:
            total_bytes += size * 4
        elif tensor.data_type == 11:
            total_bytes += size * 8
        elif tensor.data_type in {6, 7}:
            total_bytes += size * 4
        else:
            total_bytes += size * 4

    memory_mb = total_bytes / (1024 * 1024)
    return total_params, memory_mb


def test_one_benchmark(
    onnx_path: str,
    output_dir: str = "results/test_benchmarks",
    benchmark_name: str | None = None,
) -> dict:
    """Test optimization on one model.

    :param onnx_path: Path to ONNX model file
    :param output_dir: Directory to save optimized model
    :param benchmark_name: Benchmark name (auto-detected if None)
    :return: Dictionary with test results
    """
    if benchmark_name is None:
        benchmark_name = get_benchmark_name(onnx_path)
    model_name = Path(onnx_path).name
    config = get_preset(benchmark_name, model_name)
    has_batch_dim = config.has_batch_dim

    original_model = load_onnx_model(onnx_path)
    original_node_count = len(original_model.graph.node)
    original_params, original_memory = _calculate_model_parameters(original_model)

    benchmark_output_dir = Path(output_dir) / benchmark_name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = Path(onnx_path).name
    onnx_output_path = benchmark_output_dir / model_filename
    topology_output_path = benchmark_output_dir / model_filename.replace(
        ".onnx", "_topology.json"
    )

    temp_path = str(Path(onnx_path).with_suffix("")) + "_temp.onnx"
    onnx.save(original_model, temp_path)

    try:
        slimonnx = SlimONNX()

        patterns_before = slimonnx.detect_patterns(temp_path, config=config)
        total_patterns = sum(p["count"] for p in patterns_before.values())

        start_time = time.perf_counter()

        slimonnx.slim(temp_path, str(onnx_output_path), config=config)
        elapsed_time = time.perf_counter() - start_time

        optimized_model = onnx.load(str(onnx_output_path))
        optimized_node_count = len(optimized_model.graph.node)
        optimized_params, optimized_memory = _calculate_model_parameters(
            optimized_model
        )

        analyze_model(
            str(onnx_output_path),
            export_topology=True,
            topology_output_path=str(topology_output_path),
            has_batch_dim=has_batch_dim,
        )

        node_reduction = original_node_count - optimized_node_count
        node_reduction_pct = (
            (node_reduction / original_node_count * 100)
            if original_node_count > 0
            else 0.0
        )

        param_reduction = original_params - optimized_params
        memory_reduction = original_memory - optimized_memory

        return {
            "success": True,
            "benchmark": benchmark_name,
            "model": model_filename,
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
            "patterns_detail": {k: v["count"] for k, v in patterns_before.items()},
            "onnx_path": str(onnx_output_path),
            "topology_path": str(topology_output_path),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "model": model_filename,
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
            "onnx_path": None,
            "topology_path": None,
            "error": str(e),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_all_benchmarks(
    benchmark_dir: str = "benchmarks",
    output_dir: str = "results/test_benchmarks",
    max_per_benchmark: int = 20,
) -> dict:
    """Test optimization on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param output_dir: Directory to save optimized models
    :param max_per_benchmark: Maximum models per benchmark
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmark_folders(benchmark_dir)
    onnx_files = find_onnx_files_from_instances(
        benchmark_dirs, num_limit=max_per_benchmark
    )

    print(f"Testing {len(onnx_files)} models, saving to {output_dir}/")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_time = 0.0
    total_node_reduction = 0
    total_original_nodes = 0
    total_optimized_nodes = 0
    total_param_reduction = 0
    total_memory_reduction = 0.0
    total_patterns = 0

    pattern_counts = defaultdict(int)
    by_benchmark = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
            "node_reduction": 0,
        }
    )

    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = test_one_benchmark(onnx_path, output_dir, benchmark_name)

        if result["success"]:
            success_count += 1
            total_time += result["time"]
            total_node_reduction += result["node_reduction"]
            total_original_nodes += result["original_nodes"]
            total_optimized_nodes += result["optimized_nodes"]
            total_param_reduction += result["param_reduction"]
            total_memory_reduction += result["memory_reduction"]
            total_patterns += result["patterns_detected"]

            by_benchmark[benchmark_name]["count"] += 1
            by_benchmark[benchmark_name]["success"] += 1
            by_benchmark[benchmark_name]["total_time"] += result["time"]
            by_benchmark[benchmark_name]["node_reduction"] += result["node_reduction"]

            for pattern_name, count in result["patterns_detail"].items():
                pattern_counts[pattern_name] += count

            print(
                f"OK ({result['time']:.2f}s, {result['node_reduction_pct']:.1f}% reduction, {result['patterns_detected']} patterns)"
            )
        else:
            failed_count += 1
            by_benchmark[benchmark_name]["count"] += 1
            by_benchmark[benchmark_name]["failed"] += 1
            print(f"FAILED: {result['error']}")

    elapsed_total = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_count/len(onnx_files)*100:.1f}%")
    print()
    print(f"Total optimization time: {total_time:.2f}s")
    print(
        f"Avg time per model: {total_time/success_count:.2f}s"
        if success_count > 0
        else "N/A"
    )
    print(f"Total wall time: {elapsed_total:.2f}s")
    print()
    print(
        f"Node reduction: {total_original_nodes} -> {total_optimized_nodes} ({total_node_reduction} nodes)"
    )
    print(
        f"Node reduction rate: {total_node_reduction/total_original_nodes*100:.1f}%"
        if total_original_nodes > 0
        else "N/A"
    )
    print(
        f"Avg nodes reduced per model: {total_node_reduction/success_count:.1f}"
        if success_count > 0
        else "N/A"
    )
    print()
    print(f"Parameter reduction: {total_param_reduction:,} parameters")
    print(f"Memory reduction: {total_memory_reduction:.2f} MB")
    print()
    print(f"Total patterns detected: {total_patterns}")

    if pattern_counts:
        print("\nTop patterns detected:")
        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )
        for pattern_name, count in sorted_patterns[:10]:
            print(f"  {pattern_name}: {count}")

    if by_benchmark:
        print("\nBy benchmark:")
        for bname in sorted(by_benchmark.keys()):
            stats = by_benchmark[bname]
            print(
                f"  {bname}: {stats['success']}/{stats['count']} success, "
                f"{stats['total_time']:.2f}s, {stats['node_reduction']} nodes reduced"
            )

    summary = {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "total_time": total_time,
        "wall_time": elapsed_total,
        "total_node_reduction": total_node_reduction,
        "total_original_nodes": total_original_nodes,
        "total_optimized_nodes": total_optimized_nodes,
        "total_param_reduction": total_param_reduction,
        "total_memory_reduction": total_memory_reduction,
        "total_patterns": total_patterns,
        "pattern_counts": dict(pattern_counts),
        "by_benchmark": dict(by_benchmark),
    }

    summary_path = Path(output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return summary


def main() -> None:
    """Main entry point for script execution."""
    import sys

    max_models = 20
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_models = int(sys.argv[1])

    test_all_benchmarks(max_per_benchmark=max_models)


if __name__ == "__main__":
    main()
