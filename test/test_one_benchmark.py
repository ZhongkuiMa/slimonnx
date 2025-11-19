"""Generate baselines for a single benchmark."""

__docformat__ = "restructuredtext"
__all__ = ["generate_baseline", "main"]

import os
import shutil
import sys
from pathlib import Path

from benchmark_config import get_test_data_path
from slim_kwargs import SLIM_KWARGS
from slimonnx import SlimONNX
from slimonnx.slimonnx.analyze_structure import analyze_model
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
    load_onnx_model,
)


def generate_baseline(
    onnx_path: str,
    baseline_dir: str,
    verbose: bool = False,
    validate_outputs: bool = False,
) -> dict[str, int | float | str | bool]:
    """Generate baseline for a single model.

    :param onnx_path: Path to ONNX model file
    :param baseline_dir: Directory to save baseline files
    :param verbose: Whether to print verbose output during optimization
    :param validate_outputs: Whether to validate optimized outputs match original
    :return: Dictionary with results (success, time, node counts, paths)
    """
    benchmark_name = get_benchmark_name(onnx_path)
    opt_config = dict(SLIM_KWARGS[benchmark_name])
    has_batch_dim = if_has_batch_dim(onnx_path)

    if verbose:
        print(
            f"\nDEBUG: benchmark={benchmark_name}, has_batch_dim={has_batch_dim}, path={onnx_path}"
        )

    # Load original model to get node count (with version conversion to v21)
    original_model = load_onnx_model(onnx_path)
    original_node_count = len(original_model.graph.node)

    # Create temp path for optimized model
    temp_optimized_path = onnx_path.replace(".onnx", "_temp_optimized.onnx")

    try:
        # Get test data path if validation is requested
        test_data_path = None
        if validate_outputs:
            test_data_path = get_test_data_path(onnx_path, benchmark_name)
            if test_data_path and verbose:
                print(f"  Using test data from: {test_data_path}")

        # Run optimization directly on original file
        slimonnx = SlimONNX(verbose=verbose)
        report = slimonnx.slim(
            onnx_path,
            temp_optimized_path,
            has_batch_dim=has_batch_dim,
            validate_outputs=validate_outputs,
            validation_test_data_path=test_data_path,
            return_report=True,
            **opt_config,
        )

        # Extract metrics from report
        elapsed_time = report["optimization_time"]
        optimized_node_count = report["optimized_nodes"]
        reduction = report["reduction"]
        reduction_pct = report["reduction_pct"]
        validation_result = report.get("validation")

        # Save to baseline directory
        Path(baseline_dir).mkdir(parents=True, exist_ok=True)

        # Save optimized ONNX
        model_filename = Path(onnx_path).name
        onnx_output_path = Path(baseline_dir) / model_filename
        shutil.copy(temp_optimized_path, onnx_output_path)

        # Generate topology JSON
        topology_output_path = Path(baseline_dir) / model_filename.replace(
            ".onnx", "_topology.json"
        )
        analyze_model(
            str(onnx_output_path),
            verbose=False,
            export_topology=True,
            topology_output_path=str(topology_output_path),
            has_batch_dim=has_batch_dim,
        )

        result = {
            "success": True,
            "benchmark": benchmark_name,
            "time": elapsed_time,
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
            "num_optimizations": len(opt_config),
            "onnx_path": str(onnx_output_path),
            "topology_path": str(topology_output_path),
            "error": None,
        }

        # Add validation results if validation was performed
        if validation_result is not None:
            result["validation"] = {
                "all_match": validation_result["all_match"],
                "num_tests": validation_result["num_tests"],
                "passed": validation_result["passed"],
                "failed": validation_result["failed"],
                "max_diff": validation_result["max_diff"],
            }

        return result

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
            "onnx_path": None,
            "topology_path": None,
            "error": str(e),
        }

    finally:
        # Clean up temp file
        if os.path.exists(temp_optimized_path):
            os.remove(temp_optimized_path)


def main(
    benchmark_name: str,
    verbose: bool = False,
    max_models: int | None = None,
    validate_outputs: bool = False,
):
    """Generate baselines for all models in a benchmark.

    :param benchmark_name: Name of benchmark (e.g., 'acasxu_2023')
    :param verbose: Whether to print verbose output during optimization
    :param max_models: Maximum number of models to process (None = all)
    :param validate_outputs: Whether to validate optimized outputs match original
    """
    print(f"Generating baselines for benchmark: {benchmark_name}")
    print("=" * 80)

    # Find benchmark directory
    benchmark_dirs = find_benchmarks_folders("benchmarks")
    target_dir = None
    for d in benchmark_dirs:
        if benchmark_name in d:
            target_dir = d
            break

    if target_dir is None:
        print(f"Error: Benchmark '{benchmark_name}' not found in benchmarks/")
        return False

    # Find all ONNX files
    onnx_files = find_all_onnx_files([target_dir], num_limit=max_models)

    if not onnx_files:
        print(f"No ONNX files found in {target_dir}")
        return False

    print(f"Found {len(onnx_files)} models to process")
    print(f"Baseline directory: baselines/{benchmark_name}/")
    print()

    # Create baseline directory
    baseline_dir = f"baselines/{benchmark_name}"

    # Process each model
    success_count = 0
    failed_count = 0
    total_time = 0.0
    validation_passed = 0
    validation_failed = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        result = generate_baseline(
            onnx_path, baseline_dir, verbose=verbose, validate_outputs=validate_outputs
        )

        if result["success"]:
            success_count += 1
            total_time += result["time"]

            # Build output message
            msg = (
                f"OK ({result['time']:.2f}s, {result['original_nodes']}->{result['optimized_nodes']} nodes, "
                f"{result['reduction_pct']:.1f}% reduction, {result['num_optimizations']} opts"
            )

            # Add validation info if available
            if "validation" in result:
                val = result["validation"]
                if val["all_match"]:
                    validation_passed += 1
                    msg += f", validation: PASS"
                else:
                    validation_failed += 1
                    msg += f", validation: FAIL ({val['failed']}/{val['num_tests']} failed, max_diff={val['max_diff']:.2e})"

            msg += ")"
            print(msg)
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_count/len(onnx_files)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Avg time per model: {total_time/success_count:.2f}s"
        if success_count > 0
        else "N/A"
    )

    if validate_outputs:
        print()
        print(f"Validation:")
        print(f"  Passed: {validation_passed}")
        print(f"  Failed: {validation_failed}")
        if validation_passed + validation_failed > 0:
            print(
                f"  Pass rate: {validation_passed/(validation_passed+validation_failed)*100:.1f}%"
            )

    print()
    print(f"Baselines saved to: {baseline_dir}/")
    print(f"  {success_count} ONNX models")
    print(f"  {success_count} topology JSON files")

    return failed_count == 0


if __name__ == "__main__":
    # Default to acasxu_2023 if no argument provided
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "acasxu_2023"

    # Optional: max number of models
    max_models = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    # Optional flags
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    validate = "--validate" in sys.argv

    success = main(
        benchmark, verbose=verbose, max_models=max_models, validate_outputs=validate
    )
    sys.exit(0 if success else 1)
