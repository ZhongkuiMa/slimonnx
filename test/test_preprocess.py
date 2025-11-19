"""Test preprocessing functionality for SlimONNX.

Tests model loading, version conversion, and shape inference.
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_preprocess", "test_all_preprocess"]

import os
import sys
from collections import defaultdict
from pathlib import Path

from slimonnx import SlimONNX
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
)


def test_one_preprocess(
    onnx_path: str,
    target_opset: int | None = None,
    verbose: bool = False,
) -> dict:
    """Test preprocessing on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :param target_opset: Target opset version (None = keep original)
    :param verbose: Print detailed output
    :return: Preprocessing test result dictionary
    """
    slimonnx = SlimONNX(verbose=verbose)
    benchmark_name = get_benchmark_name(onnx_path)

    try:
        # Test preprocessing
        model = slimonnx.preprocess(
            onnx_path,
            target_opset=target_opset,
            infer_shapes=True,
        )

        # Extract metadata
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
    verbose: bool = False,
    target_opset: int | None = None,
    results_dir: str = "results/preprocess",
) -> bool:
    """Test preprocessing on all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param max_per_benchmark: Maximum models per benchmark to process
    :param verbose: Print detailed output
    :param target_opset: Target opset version (None = keep original)
    :param results_dir: Directory to save test results
    :return: True if all tests passed, False otherwise
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    if not onnx_files:
        print(f"No ONNX files found in {benchmark_dir}")
        return False

    print(
        f"Testing preprocessing on {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks"
    )
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_nodes": 0,
            "has_shapes_count": 0,
            "errors": [],
        }
    )

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_one_preprocess(
            onnx_path,
            target_opset=target_opset,
            verbose=verbose,
        )

        benchmark = result["benchmark"]
        stats = benchmark_stats[benchmark]
        stats["count"] += 1

        if result["success"]:
            stats["success"] += 1
            stats["total_nodes"] += result["node_count"]
            if result["has_shapes"]:
                stats["has_shapes_count"] += 1

            print(
                f"OK (opset={result['opset']}, IR={result['ir_version']}, "
                f"{result['node_count']} nodes, shapes={'yes' if result['has_shapes'] else 'no'})"
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
        shapes_pct = (
            (stats["has_shapes_count"] / stats["success"] * 100)
            if stats["success"] > 0
            else 0
        )

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Avg nodes: {avg_nodes:.0f}")
        print(
            f"  Has shapes: {stats['has_shapes_count']}/{stats['success']} ({shapes_pct:.1f}%)"
        )

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

    # Save results directory marker
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(results_dir) / "test_results.txt", "w") as f:
        f.write(f"Total models: {len(onnx_files)}\n")
        f.write(f"Success: {total_success}\n")
        f.write(f"Failed: {total_failed}\n")
        f.write(f"Success rate: {total_success/len(onnx_files)*100:.1f}%\n")

    return total_failed == 0


if __name__ == "__main__":
    success = test_all_preprocess(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,
        verbose=False,
    )
    sys.exit(0 if success else 1)
