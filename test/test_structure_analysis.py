"""Test structure analysis functionality for SlimONNX.

Tests model structure analysis, topology extraction, and reporting.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_structure_analysis", "test_all_structure_analysis"]

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx

from slimonnx import SlimONNX, OptimizationConfig
from slimonnx.slimonnx.structure_analysis.topology import build_topology
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
)


def calculate_topology_metrics(nodes: list) -> dict:
    """Calculate topology complexity metrics from ONNX nodes.

    :param nodes: List of ONNX NodeProto objects
    :return: Dictionary of topology metrics
    """
    if not nodes:
        return {
            "avg_predecessors": 0.0,
            "avg_successors": 0.0,
            "max_fan_in": 0,
            "max_fan_out": 0,
        }

    topology = build_topology(nodes)

    predecessors_counts = [len(info["predecessors"]) for info in topology.values()]
    successors_counts = [len(info["successors"]) for info in topology.values()]

    return {
        "avg_predecessors": (
            np.mean(predecessors_counts) if predecessors_counts else 0.0
        ),
        "avg_successors": np.mean(successors_counts) if successors_counts else 0.0,
        "max_fan_in": max(predecessors_counts) if predecessors_counts else 0,
        "max_fan_out": max(successors_counts) if successors_counts else 0,
    }


def calculate_shape_coverage(
    nodes: list, data_shapes: dict[str, list[int]] | None
) -> dict:
    """Calculate shape inference coverage metrics.

    :param nodes: List of ONNX NodeProto objects
    :param data_shapes: Dictionary mapping tensor names to shapes (or None)
    :return: Dictionary of shape coverage metrics
    """
    if not nodes:
        return {
            "shape_coverage_pct": 0.0,
            "nodes_with_shapes": 0,
            "nodes_missing_shapes": 0,
        }

    if data_shapes is None:
        return {
            "shape_coverage_pct": 0.0,
            "nodes_with_shapes": 0,
            "nodes_missing_shapes": len(nodes),
        }

    nodes_with_shapes = 0
    for node in nodes:
        if node.output:
            # Check if any output has a known shape
            if any(out in data_shapes for out in node.output):
                nodes_with_shapes += 1

    nodes_missing_shapes = len(nodes) - nodes_with_shapes
    coverage_pct = (nodes_with_shapes / len(nodes) * 100) if nodes else 0.0

    return {
        "shape_coverage_pct": coverage_pct,
        "nodes_with_shapes": nodes_with_shapes,
        "nodes_missing_shapes": nodes_missing_shapes,
    }


def calculate_parameter_metrics(initializers: dict[str, onnx.TensorProto]) -> dict:
    """Calculate parameter count and memory metrics.

    :param initializers: Dictionary of ONNX initializers
    :return: Dictionary of parameter metrics
    """
    if not initializers:
        return {
            "total_parameters": 0,
            "memory_mb": 0.0,
            "largest_initializer_size": 0,
        }

    total_params = 0
    total_bytes = 0
    largest_size = 0

    for tensor in initializers.values():
        # Calculate number of elements
        size = 1
        for dim in tensor.dims:
            size *= dim

        total_params += size
        largest_size = max(largest_size, size)

        # Estimate bytes based on data type
        # Most common: FLOAT (1) = 4 bytes, DOUBLE (11) = 8 bytes
        if tensor.data_type == 1:  # FLOAT
            total_bytes += size * 4
        elif tensor.data_type == 11:  # DOUBLE
            total_bytes += size * 8
        elif tensor.data_type in {6, 7}:  # INT32, INT64
            total_bytes += size * 4
        else:
            # Default to 4 bytes for other types
            total_bytes += size * 4

    memory_mb = total_bytes / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "memory_mb": memory_mb,
        "largest_initializer_size": largest_size,
    }


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

        # Load model to get nodes and initializers for new metrics
        model = onnx.load(onnx_path)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        # Calculate new metrics
        topology_metrics = calculate_topology_metrics(nodes)
        shape_metrics = calculate_shape_coverage(
            nodes, analysis.get("structure", {}).get("data_shapes")
        )
        parameter_metrics = calculate_parameter_metrics(initializers)

        return {
            "success": True,
            "benchmark": benchmark_name,
            # Basic structure metrics
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
            # Advanced metrics
            "avg_predecessors": topology_metrics["avg_predecessors"],
            "avg_successors": topology_metrics["avg_successors"],
            "max_fan_in": topology_metrics["max_fan_in"],
            "max_fan_out": topology_metrics["max_fan_out"],
            "shape_coverage_pct": shape_metrics["shape_coverage_pct"],
            "nodes_with_shapes": shape_metrics["nodes_with_shapes"],
            "nodes_missing_shapes": shape_metrics["nodes_missing_shapes"],
            "total_parameters": parameter_metrics["total_parameters"],
            "memory_mb": parameter_metrics["memory_mb"],
            "largest_initializer_size": parameter_metrics["largest_initializer_size"],
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
            "avg_predecessors": 0.0,
            "avg_successors": 0.0,
            "max_fan_in": 0,
            "max_fan_out": 0,
            "shape_coverage_pct": 0.0,
            "nodes_with_shapes": 0,
            "nodes_missing_shapes": 0,
            "total_parameters": 0,
            "memory_mb": 0.0,
            "largest_initializer_size": 0,
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
            "total_parameters": 0,
            "total_memory_mb": 0.0,
            "total_avg_predecessors": 0.0,
            "total_avg_successors": 0.0,
            "total_shape_coverage": 0.0,
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
            stats["total_parameters"] += result["total_parameters"]
            stats["total_memory_mb"] += result["memory_mb"]
            stats["total_avg_predecessors"] += result["avg_predecessors"]
            stats["total_avg_successors"] += result["avg_successors"]
            stats["total_shape_coverage"] += result["shape_coverage_pct"]
            if result["has_shapes"]:
                stats["has_shapes_count"] += 1

            top_ops_str = ", ".join(
                f"{op}={count}" for op, count in list(result["top_ops"].items())[:3]
            )
            print(
                f"OK ({result['node_count']} nodes, {result['op_types']} op types, "
                f"{result['total_parameters']:,} params, {result['memory_mb']:.1f}MB, "
                f"shape_cov={result['shape_coverage_pct']:.1f}%)"
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
        success_count = stats["success"]

        avg_nodes = stats["total_nodes"] / success_count if success_count > 0 else 0
        avg_inits = (
            stats["total_initializers"] / success_count if success_count > 0 else 0
        )
        avg_params = (
            stats["total_parameters"] / success_count if success_count > 0 else 0
        )
        avg_memory = (
            stats["total_memory_mb"] / success_count if success_count > 0 else 0
        )
        avg_predecessors = (
            stats["total_avg_predecessors"] / success_count if success_count > 0 else 0
        )
        avg_successors = (
            stats["total_avg_successors"] / success_count if success_count > 0 else 0
        )
        avg_shape_cov = (
            stats["total_shape_coverage"] / success_count if success_count > 0 else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Avg nodes: {avg_nodes:.0f}")
        print(f"  Avg initializers: {avg_inits:.0f}")
        print(f"  Avg parameters: {avg_params:,.0f}")
        print(f"  Avg memory: {avg_memory:.1f} MB")
        print(
            f"  Avg topology - predecessors: {avg_predecessors:.2f}, successors: {avg_successors:.2f}"
        )
        print(f"  Avg shape coverage: {avg_shape_cov:.1f}%")
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
