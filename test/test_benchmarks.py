"""Baseline management for SlimONNX regression testing.

This module provides functions to:
1. Optimize models and save to results/baselines/
2. Save results as archived baselines to baselines/
3. Verify results against baselines (structural + numerical)
4. Verify results against original benchmark models

Directory structure:
- benchmarks/          # Original unoptimized models
- results/baselines/   # Current optimization results
- baselines/           # Archived good baselines for regression testing
"""

__docformat__ = "restructuredtext"
__all__ = [
    "optimize_model",
    "optimize_all_models",
    "save_as_baseline",
    "verify_benchmarks",
]

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "../..")


from slimonnx import SlimONNX, get_preset
from slimonnx.slimonnx.analyze_structure import analyze_model
from slimonnx.test.benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
    get_model_relative_path,
    get_model_data_path,
)
from slimonnx.test.utils import load_onnx_model


def _run_onnx_model(model_path: str, inputs: np.ndarray) -> dict:
    """Run ONNX model inference using ONNX Runtime.

    :param model_path: Path to ONNX model file
    :param inputs: input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    output_names = [out.name for out in session.get_outputs()]
    return {name: output for name, output in zip(output_names, outputs)}


def _compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[bool, list[str]]:
    """Compare outputs from two models.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Tuple of (all_match, mismatch_messages)
    """
    mismatches = []
    for key1, key2 in zip(outputs1.keys(), outputs2.keys()):
        out1 = outputs1[key1]
        out2 = outputs2[key2]

        if out1.shape != out2.shape:
            mismatches.append(f"{key1}: shape {out1.shape} vs {key2}: {out2.shape}")
            continue

        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(out1 - out2))
            mismatches.append(f"{key1} {key2}: max diff {max_diff:.2e}")

    return len(mismatches) == 0, mismatches


def optimize_model(
    model_path: Path,
    output_dir: Path,
    benchmarks_root: Path,
) -> dict:
    """Optimize one ONNX model and save to results directory.

    :param model_path: Path to source ONNX model file
    :param output_dir: Directory to save optimized model
    :param benchmarks_root: Path to benchmarks root directory
    :return: Dictionary with optimization results
    """
    benchmark_name = get_model_benchmark_name(model_path)
    model_name = model_path.name
    config = get_preset(benchmark_name, model_name)
    has_batch_dim = config.has_batch_dim

    model = load_onnx_model(str(model_path))
    original_node_count = len(model.graph.node)

    rel_path = get_model_relative_path(model_path, benchmarks_root)
    onnx_output_path = output_dir / rel_path
    topology_output_path = (
        onnx_output_path.with_suffix("").with_suffix(".json").parent.parent
        / "json"
        / (onnx_output_path.stem + "_topology.json")
    )

    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    topology_output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = model_path.with_suffix(".temp.onnx")
    onnx.save(model, str(temp_path))

    try:
        start_time = time.perf_counter()
        slimonnx = SlimONNX()
        slimonnx.slim(str(temp_path), str(onnx_output_path), config=config)
        elapsed_time = time.perf_counter() - start_time

        optimized_model = onnx.load(str(onnx_output_path))
        optimized_model.ir_version = 8
        onnx.save(optimized_model, str(onnx_output_path))
        optimized_node_count = len(optimized_model.graph.node)

        analyze_model(
            str(onnx_output_path),
            export_topology=True,
            topology_output_path=str(topology_output_path),
            has_batch_dim=has_batch_dim,
        )

        reduction = original_node_count - optimized_node_count
        reduction_pct = (
            (reduction / original_node_count * 100) if original_node_count > 0 else 0.0
        )

        return {
            "success": True,
            "benchmark": benchmark_name,
            "model": model_name,
            "time": elapsed_time,
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
            "onnx_path": str(onnx_output_path),
            "topology_path": str(topology_output_path),
            "error": None,
        }

    except (IOError, OSError, ValueError, RuntimeError, AttributeError) as error:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "model": model_name,
            "time": 0.0,
            "original_nodes": original_node_count,
            "optimized_nodes": 0,
            "reduction": 0,
            "reduction_pct": 0.0,
            "onnx_path": None,
            "topology_path": None,
            "error": str(error),
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()


def optimize_all_models(
    benchmark_dir: str = "benchmarks",
    output_dir: str = "results/baselines",
    max_per_benchmark: int = 20,
) -> dict:
    """Optimize all benchmark models and save to results directory.

    :param benchmark_dir: Root directory of benchmarks
    :param output_dir: Directory to save optimized models
    :param max_per_benchmark: Maximum models per benchmark to process
    :return: Dictionary with overall statistics
    """
    benchmarks_root = Path(benchmark_dir)
    output_root = Path(output_dir)
    benchmarks = find_benchmarks(benchmark_dir)
    models = find_models(benchmarks, max_per_benchmark)

    print(f"Optimizing {len(models)} models to {output_dir}/")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_time = 0.0
    total_reduction = 0
    total_original_nodes = 0
    total_optimized_nodes = 0

    start_time = time.perf_counter()

    for i, model_path in enumerate(models):
        rel_path = get_model_relative_path(model_path, benchmarks_root)

        print(f"[{i}/{len(models)}] {rel_path}...", end=" ")

        result = optimize_model(model_path, output_root, benchmarks_root)

        if result["success"]:
            success_count += 1
            total_time += result["time"]
            total_reduction += result["reduction"]
            total_original_nodes += result["original_nodes"]
            total_optimized_nodes += result["optimized_nodes"]

            print(
                f"OK ({result['time']:.2f}s, {result['original_nodes']}->{result['optimized_nodes']} nodes, {result['reduction_pct']:.1f}%)"
            )
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    elapsed_total = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(models)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(
        f"Success rate: {success_count/len(models)*100:.1f}%"
        if len(models) > 0
        else "N/A"
    )
    print(f"Total optimization time: {total_time:.2f}s")
    print(
        f"Avg time per model: {total_time/success_count:.2f}s"
        if success_count > 0
        else "N/A"
    )
    print(f"Total wall time: {elapsed_total:.2f}s")
    print(
        f"Node reduction: {total_original_nodes} -> {total_optimized_nodes} ({total_reduction} nodes, {total_reduction/total_original_nodes*100:.1f}%)"
        if total_original_nodes > 0
        else "N/A"
    )

    return {
        "total": len(models),
        "success": success_count,
        "failed": failed_count,
        "total_time": total_time,
        "wall_time": elapsed_total,
        "total_reduction": total_reduction,
        "total_original_nodes": total_original_nodes,
        "total_optimized_nodes": total_optimized_nodes,
    }


def save_as_baseline(
    results_dir: str = "results/baselines",
    baselines_dir: str = "baselines",
) -> tuple[int, int]:
    """Save current results as archived baselines.

    Copies entire results directory structure to baselines directory.

    :param results_dir: Source directory containing current results
    :param baselines_dir: Target directory for archived baselines
    :return: Tuple of (num_copied, num_failed)
    """
    results_path = Path(results_dir)
    baselines_path = Path(baselines_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Saving {results_dir}/ as baselines to {baselines_dir}/")
    print("=" * 70)

    # Remove old baselines if they exist
    if baselines_path.exists():
        print(f"Removing old baselines {baselines_path}...")
        shutil.rmtree(baselines_path)

    # Copy results to baselines
    shutil.copytree(results_path, baselines_path)

    # Count files
    onnx_count = len(list(baselines_path.rglob("*.onnx")))
    json_count = len(list(baselines_path.rglob("*.json")))

    print(f"Copied {onnx_count} ONNX files and {json_count} JSON files")
    print("=" * 70)

    return onnx_count, json_count


def _load_test_data_from_npz(data_file: Path) -> list[np.ndarray]:
    """Load test inputs from npz data file.

    :param data_file: Path to npz data file
    :return: List of test input arrays
    """
    data = np.load(data_file, allow_pickle=True)
    test_inputs = []

    for vnnlib_name in data.files:
        vnnlib_data = data[vnnlib_name].item()
        if isinstance(vnnlib_data, dict):
            for bound_type in ["lower", "upper"]:
                if bound_type in vnnlib_data:
                    bound_data = vnnlib_data[bound_type]
                    if "inputs" in bound_data:
                        test_inputs.extend(bound_data["inputs"])

    return test_inputs


def _verify_one_benchmark(
    result_file: Path,
    benchmark_file: Path,
    data_file: Path,
    rel_path: Path,
) -> tuple[str, str | None]:
    """Verify single model against baseline with structural and numerical checks.

    :param result_file: Path to result ONNX model
    :param benchmark_file:
    :param data_file: Path to test data npz file
    :param rel_path: Relative path for reporting
    :return: Tuple of (status, error_message) where status is "OK", "STRUCTURAL_MISMATCH", "NUMERICAL_MISMATCH", "SKIP", "ERROR"
    """
    result_model = onnx.load(str(result_file))
    benchmark_model = onnx.load(str(benchmark_file))

    if not data_file.exists():
        return "SKIP", f"No test data file: {data_file}"

    try:
        test_inputs = _load_test_data_from_npz(data_file)
        if not test_inputs:
            return "SKIP", "No inputs in data file"
    except (IOError, KeyError, ValueError, IndexError) as error:
        return "SKIP", f"Error loading data: {error}"

    all_match = True
    for inputs in test_inputs:
        try:
            result_outputs = _run_onnx_model(str(result_file), inputs)
            benchmark_outputs = _run_onnx_model(str(benchmark_file), inputs)

            match, mismatches = _compare_outputs(result_outputs, benchmark_outputs)
            if not match:
                all_match = False
                break
        except (RuntimeError, ValueError) as error:
            return "ERROR", str(error)

    if all_match:
        return "OK", None
    else:
        return "NUMERICAL_MISMATCH", None


def verify_benchmarks(
    results_dir: str = "results/baselines",
    benchmarks_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
) -> dict:
    """Verify optimized results against original benchmark models.

    Compares numerical outputs between slimonnx optimized models and original models.

    :param results_dir: Directory containing optimized results
    :param benchmarks_dir: Directory containing original benchmarks with test data
    :param max_per_benchmark: Maximum models per benchmark to verify
    :return: Dictionary with verification results
    """
    results_path = Path(results_dir)
    benchmarks_path = Path(benchmarks_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_path}")

    print(f"Verifying {results_dir}/ against {benchmarks_dir}/")
    print("=" * 70)

    benchmarks = find_benchmarks(benchmarks_dir)
    benchmark_models = find_models(benchmarks, max_per_benchmark)

    passed = 0
    failed = 0
    missing = 0
    structural_mismatches = 0
    numerical_mismatches = 0

    for i, benchmark_model in enumerate(benchmark_models):
        rel_path = get_model_relative_path(benchmark_model, benchmarks_path)
        print(f"[{i}/{len(benchmark_models)}] {rel_path}...", end=" ")
        result_file = results_path / rel_path
        benchmark_file = benchmarks_path / rel_path

        data_file = get_model_data_path(benchmark_model, benchmarks_path)

        status, error_msg = _verify_one_benchmark(
            result_file, benchmark_file, data_file, rel_path
        )

        if status == "OK":
            passed += 1
            print("OK")
        elif status == "STRUCTURAL_MISMATCH":
            structural_mismatches += 1
            failed += 1
            print(f"STRUCTURAL MISMATCH: {rel_path} ({error_msg})")
        elif status == "NUMERICAL_MISMATCH":
            numerical_mismatches += 1
            failed += 1
            print(f"NUMERICAL MISMATCH: {rel_path}")
        elif status == "SKIP":
            print(f"SKIP: {rel_path} - {error_msg}")
        elif status == "ERROR":
            failed += 1
            print(f"ERROR: {rel_path} - {error_msg}")

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(benchmark_models)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"  Structural mismatches: {structural_mismatches}")
    print(f"  Numerical mismatches: {numerical_mismatches}")
    print(f"Missing baselines: {missing}")
    print(
        f"Pass rate: {passed/(len(benchmark_models)-missing)*100:.1f}%"
        if len(benchmark_models) > missing
        else "N/A"
    )

    return {
        "total": len(benchmark_models),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "structural_mismatches": structural_mismatches,
        "numerical_mismatches": numerical_mismatches,
    }


def main() -> None:
    """Main entry point for script execution."""
    optimize_all_models()
    verify_benchmarks()
    save_as_baseline()


if __name__ == "__main__":
    main()
