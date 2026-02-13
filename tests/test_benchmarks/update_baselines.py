#!/usr/bin/env python3
"""Update baseline outputs for SlimONNX regression testing.

This script copies optimization statistics from results/ to baselines/ to update
golden references for regression testing.

Workflow:
    1. Run test_slimonnx.py to generate results/
    2. Run this script to copy results/ -> baselines/
    3. Run test_slimonnx_regression.py to verify

Usage:
    python update_baselines.py                     # Copy all results to baselines
    python update_baselines.py --benchmark acasxu_2023  # Copy specific benchmark
    python update_baselines.py --dry-run           # Show what would be copied
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_baseline(
    result_path: Path, baseline_path: Path, dry_run: bool = False
) -> tuple[bool, int]:
    """Copy a single result JSON to baseline.

    :param result_path: Source .json file in results/
    :param baseline_path: Destination .json file in baselines/
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success, file_count)
    """
    if dry_run:
        if result_path.exists():
            print(f"  [DRY-RUN] Would copy: {result_path} -> {baseline_path}")
            return True, 1
        print(f"  [DRY-RUN] Would skip (missing): {result_path}")
        return False, 0

    # Check source exists
    if not result_path.exists():
        print(f"  [SKIP] Missing result: {result_path}")
        return False, 0

    # Create baseline directory
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy result to baseline
    shutil.copy2(result_path, baseline_path)
    print(f"  [OK] Copied: {baseline_path}")
    return True, 1


def update_benchmark(
    benchmark_name: str,
    results_dir: Path,
    baselines_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Update baselines for one benchmark.

    :param benchmark_name: Name of benchmark
    :param results_dir: Root results directory
    :param baselines_dir: Root baselines directory
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success_count, total_count)
    """
    result_bench_dir = results_dir / benchmark_name
    baseline_bench_dir = baselines_dir / benchmark_name

    if not result_bench_dir.exists():
        print(f"  [SKIP] No results: {benchmark_name}")
        return 0, 0

    # Copy each .json file
    success = 0
    total = 0

    for result_json in sorted(result_bench_dir.glob("*.json")):
        total += 1
        baseline_json = baseline_bench_dir / result_json.name
        copied, _ = copy_baseline(result_json, baseline_json, dry_run)
        if copied:
            success += 1

    return success, total


def get_benchmarks_to_update(benchmark_arg: str | None, results_dir: Path) -> list[str] | None:
    """Determine which benchmarks to update.

    :param benchmark_arg: Benchmark name from CLI arg (or None for all)
    :param results_dir: Root results directory
    :return: List of benchmark names to update, or None on error
    """
    if benchmark_arg:
        benchmark_path = results_dir / benchmark_arg
        if not benchmark_path.exists():
            print(f"Error: Benchmark '{benchmark_arg}' not found in results/")
            print(f"\nAvailable benchmarks in {results_dir}:")
            for bench in sorted(results_dir.iterdir()):
                if bench.is_dir():
                    print(f"  - {bench.name}")
            return None
        return [benchmark_arg]

    benchmarks = [bench.name for bench in sorted(results_dir.iterdir()) if bench.is_dir()]
    return benchmarks or None


def main():
    """Update baselines from results directory."""
    parser = argparse.ArgumentParser(
        description="Update baselines by copying .json stats from results/ to baselines/",
        epilog="Example: python update_baselines.py --benchmark acasxu_2023",
    )
    parser.add_argument(
        "--benchmark",
        help="Update specific benchmark only (e.g., acasxu_2023)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without making changes",
    )
    args = parser.parse_args()

    # Setup paths
    test_dir = Path(__file__).parent
    results_dir = test_dir / "results"
    baselines_dir = test_dir / "baselines"

    # Check results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nRun test_slimonnx.py first to generate results/")
        return 1

    # Create baselines directory if missing
    if not args.dry_run:
        baselines_dir.mkdir(exist_ok=True)

    # Find benchmarks to update
    benchmarks_to_update = get_benchmarks_to_update(args.benchmark, results_dir)
    if benchmarks_to_update is None:
        print("No benchmarks found in results/")
        return 1

    # Update baselines
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Updating baselines from results/")
    print(f"Benchmarks: {len(benchmarks_to_update)}")
    print("=" * 80)

    total_success = 0
    total_count = 0

    for benchmark_name in benchmarks_to_update:
        print(f"\n{benchmark_name}:")
        success, count = update_benchmark(
            benchmark_name, results_dir, baselines_dir, dry_run=args.dry_run
        )
        total_success += success
        total_count += count

        if count > 0:
            print(f"  {success}/{count} files copied")

    print("\n" + "=" * 80)
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Baseline update complete!")
    print(f"Total: {total_success}/{total_count} files copied")

    if not args.dry_run and total_success > 0:
        print(f"\nBaselines updated in: {baselines_dir}")
        print("Run test_slimonnx_regression.py to verify")

    return 0


if __name__ == "__main__":
    sys.exit(main())
