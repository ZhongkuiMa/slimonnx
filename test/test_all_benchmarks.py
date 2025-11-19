"""Run structural optimization on all benchmarks."""

__docformat__ = "restructuredtext"
__all__ = ["main"]

import sys
from pathlib import Path

from test_one_benchmark import main as test_benchmark


def main():
    """Run optimization on all available benchmarks.

    :return: True if all benchmarks succeeded, False otherwise
    """
    # Find all benchmark directories
    benchmarks_dir = Path("benchmarks")
    if not benchmarks_dir.exists():
        print(f"Error: Benchmarks directory not found: {benchmarks_dir}")
        return False

    # Get all benchmark directories
    benchmarks = sorted([d.name for d in benchmarks_dir.iterdir() if d.is_dir()])

    print(f"Found {len(benchmarks)} benchmarks to process")
    print("=" * 80)

    # Track results
    results = {}
    total_success = 0
    total_failed = 0

    # Process each benchmark
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] Processing {benchmark}...")
        print("-" * 80)

        try:
            # Run optimization (max 20 models, no validation)
            success = test_benchmark(
                benchmark_name=benchmark,
                verbose=False,
                max_models=20,
                validate_outputs=False,
            )

            results[benchmark] = "SUCCESS" if success else "PARTIAL"
            if success:
                total_success += 1
            else:
                total_failed += 1

        except Exception as e:
            print(f"FAILED: {benchmark}: {e}")
            results[benchmark] = f"FAILED: {e}"
            total_failed += 1

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total benchmarks: {len(benchmarks)}")
    print(f"Successful: {total_success}")
    print(f"Failed/Partial: {total_failed}")
    print(f"Success rate: {total_success/len(benchmarks)*100:.1f}%")
    print()

    # Print per-benchmark results
    print("Per-benchmark results:")
    for benchmark, result in results.items():
        status = "OK" if result == "SUCCESS" else "FAIL"
        print(f"  {status:4s} {benchmark:40s} {result}")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
