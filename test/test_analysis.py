"""Test pattern detection and structure analysis for SlimONNX.

Combines two types of model analysis:
1. Pattern Detection - Find optimization patterns (fusion, redundant operations)
2. Structure Analysis - Analyze topology, shapes, and complexity metrics
"""

__docformat__ = "restructuredtext"
__all__ = [
    "run_pattern_detection_test",
    "test_all_pattern_detection",
    "run_structure_analysis_test",
    "test_all_structure_analysis",
    "main",
]

from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx

from slimonnx import SlimONNX, OptimizationConfig
from slimonnx.slimonnx.structure_analysis.topology import build_topology
from slimonnx.test.benchmark_utils import (
    find_onnx_files_from_instances,
    find_benchmark_folders,
    get_benchmark_name,
)
from slimonnx.test.utils import if_has_batch_dim


def run_pattern_detection_test(onnx_path: str) -> dict:
    """Run pattern detection test on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :return: Pattern detection result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    config = OptimizationConfig(has_batch_dim=has_batch_dim)
    slimonnx = SlimONNX()

    try:
        patterns = slimonnx.detect_patterns(onnx_path, config=config)

        fusion_count = sum(
            p["count"] for p in patterns.values() if p["category"] == "fusion"
        )
        redundant_count = sum(
            p["count"] for p in patterns.values() if p["category"] == "redundant"
        )
        total_count = fusion_count + redundant_count

        pattern_summary = {
            name: data["count"] for name, data in patterns.items() if data["count"] > 0
        }

        return {
            "success": True,
            "benchmark": benchmark_name,
            "fusion_patterns": fusion_count,
            "redundant_patterns": redundant_count,
            "total_patterns": total_count,
            "pattern_details": pattern_summary,
            "error": None,
        }

    except (ImportError, ValueError, AttributeError, RuntimeError) as error:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "fusion_patterns": 0,
            "redundant_patterns": 0,
            "total_patterns": 0,
            "pattern_details": {},
            "error": str(error),
        }


def test_all_pattern_detection(
    benchmark_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> dict:
    """Test pattern detection on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmark_folders(benchmark_dir)
    onnx_files = find_onnx_files_from_instances(
        benchmark_dirs, num_limit=max_per_benchmark
    )

    print(f"Testing pattern detection on {len(onnx_files)} models")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_fusion = 0
    total_redundant = 0
    pattern_counts = defaultdict(int)

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = Path(onnx_path).name
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = run_pattern_detection_test(onnx_path)

        if result["success"]:
            success_count += 1
            total_fusion += result["fusion_patterns"]
            total_redundant += result["redundant_patterns"]

            for pattern_name, count in result["pattern_details"].items():
                pattern_counts[pattern_name] += count

            print(
                f"OK ({result['total_patterns']} patterns: {result['fusion_patterns']} fusion, {result['redundant_patterns']} redundant)"
            )
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    print("\n" + "=" * 70)
    print("PATTERN DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()
    print(f"Total fusion patterns: {total_fusion}")
    print(f"Total redundant patterns: {total_redundant}")
    print(f"Total patterns: {total_fusion + total_redundant}")

    if pattern_counts:
        print("\nPattern breakdown:")
        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )
        for pattern_name, count in sorted_patterns:
            print(f"  {pattern_name}: {count}")

    return {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "total_fusion": total_fusion,
        "total_redundant": total_redundant,
        "pattern_counts": dict(pattern_counts),
    }


def run_structure_analysis_test(onnx_path: str) -> dict:
    """Run structure analysis test on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :return: Structure analysis result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    try:
        model = onnx.load(onnx_path)

        node_count = len(model.graph.node)
        initializer_count = len(model.graph.initializer)

        inputs = [
            inp
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        ]
        outputs = list(model.graph.output)

        op_types = defaultdict(int)
        for node in model.graph.node:
            op_types[node.op_type] += 1

        try:
            from shapeonnx import infer_onnx_shape
            from shapeonnx.shapeonnx.utils import (
                get_initializers,
                get_input_nodes,
                get_output_nodes,
                convert_constant_to_initializer,
            )

            initializers = get_initializers(model)
            input_nodes = get_input_nodes(model, initializers, has_batch_dim)
            output_nodes = get_output_nodes(model, has_batch_dim)
            nodes = list(model.graph.node)
            nodes = convert_constant_to_initializer(nodes, initializers)

            data_shapes = infer_onnx_shape(
                input_nodes, output_nodes, nodes, initializers, has_batch_dim, False
            )
            has_shapes = True
        except (
            ImportError,
            ValueError,
            AttributeError,
            KeyError,
            RuntimeError,
        ) as error:
            print(f"Shape inference failed: {error}")
            data_shapes = None
            has_shapes = False

        topology = build_topology(model.graph.node)
        predecessors_counts = [len(info["predecessors"]) for info in topology.values()]
        successors_counts = [len(info["successors"]) for info in topology.values()]

        topology_metrics = {
            "avg_predecessors": (
                float(np.mean(predecessors_counts)) if predecessors_counts else 0.0
            ),
            "avg_successors": (
                float(np.mean(successors_counts)) if successors_counts else 0.0
            ),
            "max_fan_in": max(predecessors_counts) if predecessors_counts else 0,
            "max_fan_out": max(successors_counts) if successors_counts else 0,
        }

        nodes_with_shapes = 0
        if data_shapes:
            for node in model.graph.node:
                if any(out in data_shapes for out in node.output):
                    nodes_with_shapes += 1

        shape_coverage_pct = (
            (nodes_with_shapes / node_count * 100) if node_count > 0 else 0.0
        )

        total_parameters = 0
        total_bytes = 0
        for tensor in model.graph.initializer:
            size = 1
            for dim in tensor.dims:
                size *= dim
            total_parameters += size

            if tensor.data_type == 1:
                total_bytes += size * 4
            elif tensor.data_type == 11:
                total_bytes += size * 8
            else:
                total_bytes += size * 4

        memory_mb = total_bytes / (1024 * 1024)

        sorted_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)
        top_ops = dict(sorted_ops[:10])

        return {
            "success": True,
            "benchmark": benchmark_name,
            "node_count": node_count,
            "initializer_count": initializer_count,
            "num_inputs": len(inputs),
            "num_outputs": len(outputs),
            "has_shapes": has_shapes,
            "op_types": len(op_types),
            "top_ops": top_ops,
            "avg_predecessors": topology_metrics["avg_predecessors"],
            "avg_successors": topology_metrics["avg_successors"],
            "max_fan_in": topology_metrics["max_fan_in"],
            "max_fan_out": topology_metrics["max_fan_out"],
            "shape_coverage_pct": shape_coverage_pct,
            "nodes_with_shapes": nodes_with_shapes,
            "nodes_missing_shapes": node_count - nodes_with_shapes,
            "total_parameters": total_parameters,
            "memory_mb": memory_mb,
            "error": None,
        }

    except (IOError, ValueError, AttributeError) as error:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "error": str(error),
        }


def test_all_structure_analysis(
    benchmark_dir: str = "benchmarks", max_per_benchmark: int = 20
) -> dict:
    """Test structure analysis on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark
    :return: Dictionary with overall statistics
    """
    benchmark_dirs = find_benchmark_folders(benchmark_dir)
    onnx_files = find_onnx_files_from_instances(
        benchmark_dirs, num_limit=max_per_benchmark
    )

    print(f"Testing structure analysis on {len(onnx_files)} models")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_nodes = 0
    total_params = 0
    total_memory = 0.0
    models_with_shapes = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = Path(onnx_path).name
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = run_structure_analysis_test(onnx_path)

        if result["success"]:
            success_count += 1
            total_nodes += result["node_count"]
            total_params += result["total_parameters"]
            total_memory += result["memory_mb"]
            if result["has_shapes"]:
                models_with_shapes += 1

            print(
                f"OK ({result['node_count']} nodes, {result['op_types']} op types, "
                f"{result['shape_coverage_pct']:.1f}% shape coverage)"
            )
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    print("\n" + "=" * 70)
    print("STRUCTURE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()
    print(f"Total nodes: {total_nodes}")
    print(
        f"Avg nodes per model: {total_nodes/success_count:.1f}"
        if success_count > 0
        else "N/A"
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Total memory: {total_memory:.2f} MB")
    print(
        f"Models with shape inference: {models_with_shapes}/{success_count} ({models_with_shapes/success_count*100:.1f}%)"
        if success_count > 0
        else "N/A"
    )

    return {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "total_nodes": total_nodes,
        "total_params": total_params,
        "total_memory": total_memory,
        "models_with_shapes": models_with_shapes,
    }


def main() -> None:
    """Main entry point for script execution."""
    import sys

    if "--patterns-only" in sys.argv:
        test_all_pattern_detection()
    elif "--structure-only" in sys.argv:
        test_all_structure_analysis()
    else:
        print("Running both pattern detection and structure analysis...\n")
        test_all_pattern_detection()
        print("\n")
        test_all_structure_analysis()


if __name__ == "__main__":
    main()
