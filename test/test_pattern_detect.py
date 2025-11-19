"""Test pattern detection functionality for SlimONNX.

Tests detection of optimization patterns (fusion and redundant operations).
"""

__docformat__ = "restructuredtext"
__all__ = ["test_one_pattern_detect", "test_all_pattern_detect"]

import os
import sys
from collections import defaultdict
from pathlib import Path

from slimonnx import SlimONNX, OptimizationConfig
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
)


def test_one_pattern_detect(
    onnx_path: str,
    verbose: bool = False,
) -> dict:
    """Test pattern detection on a single ONNX model.

    :param onnx_path: Path to ONNX model file
    :param verbose: Print detailed output
    :return: Pattern detection test result dictionary
    """
    benchmark_name = get_benchmark_name(onnx_path)
    has_batch_dim = if_has_batch_dim(onnx_path)

    config = OptimizationConfig(has_batch_dim=has_batch_dim)
    slimonnx = SlimONNX(verbose=verbose)

    try:
        # Detect patterns
        patterns = slimonnx.detect_patterns(onnx_path, config=config)

        # Count patterns by category
        fusion_count = sum(
            p["count"] for p in patterns.values() if p["category"] == "fusion"
        )
        redundant_count = sum(
            p["count"] for p in patterns.values() if p["category"] == "redundant"
        )
        total_count = fusion_count + redundant_count

        # Collect pattern details
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

    except Exception as e:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "fusion_patterns": 0,
            "redundant_patterns": 0,
            "total_patterns": 0,
            "pattern_details": {},
            "error": str(e),
        }


def test_all_pattern_detect(
    benchmark_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
    verbose: bool = False,
    results_dir: str = "results/pattern_detect",
) -> bool:
    """Test pattern detection on all benchmark models.

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
        f"Testing pattern detection on {len(onnx_files)} models from {len(benchmark_dirs)} benchmarks"
    )
    print("=" * 80)

    # Track statistics by benchmark
    benchmark_stats = defaultdict(
        lambda: {
            "count": 0,
            "success": 0,
            "failed": 0,
            "total_fusion": 0,
            "total_redundant": 0,
            "total_patterns": 0,
            "pattern_counts": defaultdict(int),
            "errors": [],
        }
    )

    # Process each model
    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = test_one_pattern_detect(onnx_path, verbose=verbose)

        benchmark = result["benchmark"]
        stats = benchmark_stats[benchmark]
        stats["count"] += 1

        if result["success"]:
            stats["success"] += 1
            stats["total_fusion"] += result["fusion_patterns"]
            stats["total_redundant"] += result["redundant_patterns"]
            stats["total_patterns"] += result["total_patterns"]

            # Aggregate pattern counts
            for pattern_name, count in result["pattern_details"].items():
                stats["pattern_counts"][pattern_name] += count

            pattern_str = ", ".join(
                f"{name}={count}" for name, count in result["pattern_details"].items()
            )
            if pattern_str:
                print(f"OK ({pattern_str})")
            else:
                print("OK (no patterns)")
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

        print(f"\n{benchmark}:")
        print(
            f"  Models: {stats['count']} ({stats['success']} success, {stats['failed']} failed)"
        )
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"    Fusion: {stats['total_fusion']}")
        print(f"    Redundant: {stats['total_redundant']}")

        if stats["pattern_counts"]:
            print("  Pattern breakdown:")
            for pattern_name, count in sorted(stats["pattern_counts"].items()):
                print(f"    {pattern_name}: {count}")

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
    success = test_all_pattern_detect(
        benchmark_dir="benchmarks",
        max_per_benchmark=20,
        verbose=False,
    )
    sys.exit(0 if success else 1)
