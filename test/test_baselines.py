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
    "verify_against_baseline",
    "verify_against_benchmarks",
]

import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from slimonnx import SlimONNX, get_preset
from slimonnx.slimonnx.analyze_structure import analyze_model
from slimonnx.test.utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    load_onnx_model,
    load_vnnlib_inputs,
)


def _generate_random_inputs(model: onnx.ModelProto, num_samples: int = 5) -> list[dict]:
    """Generate random test inputs matching model input signature.

    :param model: ONNX ModelProto
    :param num_samples: Number of random input samples to generate
    :return: List of input dictionaries
    """
    inputs_list = []

    for _ in range(num_samples):
        inputs = {}
        for inp in model.graph.input:
            if any(init.name == inp.name for init in model.graph.initializer):
                continue

            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)

            dtype_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT64: np.int64,
            }
            dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)

            if dtype in [np.float32, np.float64]:
                inputs[inp.name] = np.random.randn(*shape).astype(dtype)
            else:
                inputs[inp.name] = np.random.randint(0, 10, size=shape, dtype=dtype)

        inputs_list.append(inputs)

    return inputs_list


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
            raise ValueError(f"Outputs do not match with {out1} and {out2}")

    return len(mismatches) == 0, mismatches


def optimize_model(
    onnx_path: str,
    output_dir: str = "results/baselines",
    benchmark_name: str | None = None,
) -> dict:
    """Optimize one ONNX model and save to results directory.

    :param onnx_path: Path to source ONNX model file
    :param output_dir: Directory to save optimized model (e.g., results/baselines)
    :param benchmark_name: Benchmark name (auto-detected if None)
    :return: Dictionary with optimization results
    """
    if benchmark_name is None:
        benchmark_name = get_benchmark_name(onnx_path)
    model_name = Path(onnx_path).name
    config = get_preset(benchmark_name, model_name)
    has_batch_dim = config.has_batch_dim

    model = load_onnx_model(onnx_path)
    original_node_count = len(model.graph.node)

    benchmark_output_dir = Path(output_dir) / benchmark_name
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = Path(onnx_path).name
    onnx_output_path = benchmark_output_dir / model_filename
    topology_output_path = benchmark_output_dir / model_filename.replace(
        ".onnx", "_topology.json"
    )

    temp_path = str(Path(onnx_path).with_suffix("")) + "_temp.onnx"
    onnx.save(model, temp_path)

    try:
        start_time = time.perf_counter()
        slimonnx = SlimONNX()
        slimonnx.slim(temp_path, str(onnx_output_path), config=config)
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
            "model": model_filename,
            "time": elapsed_time,
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
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
            "reduction": 0,
            "reduction_pct": 0.0,
            "onnx_path": None,
            "topology_path": None,
            "error": str(e),
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)

    print(f"Optimizing {len(onnx_files)} models to {output_dir}/")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_time = 0.0
    total_reduction = 0
    total_original_nodes = 0
    total_optimized_nodes = 0

    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = os.path.basename(onnx_path)
        benchmark_name = get_benchmark_name(onnx_path)

        print(f"[{i}/{len(onnx_files)}] {benchmark_name}/{basename}...", end=" ")

        result = optimize_model(onnx_path, output_dir, benchmark_name)

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
    print(f"Total models: {len(onnx_files)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_count/len(onnx_files)*100:.1f}%")
    print(f"Total optimization time: {total_time:.2f}s")
    print(
        f"Avg time per model: {total_time/success_count:.2f}s"
        if success_count > 0
        else "N/A"
    )
    print(f"Total wall time: {elapsed_total:.2f}s")
    print(
        f"Node reduction: {total_original_nodes} -> {total_optimized_nodes} ({total_reduction} nodes, {total_reduction/total_original_nodes*100:.1f}%)"
    )

    return {
        "total": len(onnx_files),
        "success": success_count,
        "failed": failed_count,
        "total_time": total_time,
        "wall_time": elapsed_total,
        "total_reduction": total_reduction,
        "total_original_nodes": total_original_nodes,
        "total_optimized_nodes": total_optimized_nodes,
    }


def save_as_baseline(
    results_dir: str = "results/baselines", baselines_dir: str = "baselines"
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
        print("Removing old baselines...")
        shutil.rmtree(baselines_path)

    # Copy results to baselines
    shutil.copytree(results_path, baselines_path)

    # Count files
    onnx_count = len(list(baselines_path.rglob("*.onnx")))
    json_count = len(list(baselines_path.rglob("*.json")))

    print(f"Copied {onnx_count} ONNX files and {json_count} JSON files")
    print("=" * 70)

    return onnx_count, json_count


def verify_against_baseline(
    results_dir: str = "results/baselines",
    benchmarks_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    use_vnnlib_inputs: bool = True,
    num_samples: int = 5,
) -> dict:
    """Verify current results against archived baselines.

    Compares structure (node count) and numerical outputs.

    :param results_dir: Directory containing current results
    :param baselines_dir: Directory containing archived baselines
    :param use_vnnlib_inputs: Whether to use vnnlib inputs if available
    :param num_samples: Number of random samples for testing
    :return: Dictionary with verification results
    """
    results_path = Path(results_dir)
    baselines_path = Path(baselines_dir)
    benchmarks_path = Path(benchmarks_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not baselines_path.exists():
        raise FileNotFoundError(f"Baselines directory not found: {baselines_dir}")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")
    print(f"Verifying {results_dir}/ against {baselines_dir}/")
    print("=" * 70)

    results_onnx = sorted(results_path.rglob("*.onnx"))
    baselines_onnx = {
        f.relative_to(baselines_path): f for f in baselines_path.rglob("*.onnx")
    }

    passed = 0
    failed = 0
    missing = 0
    structural_mismatches = 0
    numerical_mismatches = 0

    for result_file in results_onnx:
        rel_path = result_file.relative_to(results_path)
        baseline_file = baselines_onnx.get(rel_path)

        if baseline_file is None:
            print(f"MISSING: {rel_path} (no baseline)")
            missing += 1
            continue

        # Load models
        result_model = onnx.load(str(result_file))
        baseline_model = onnx.load(str(baseline_file))

        # Structural check
        result_nodes = len(result_model.graph.node)
        baseline_nodes = len(baseline_model.graph.node)

        if result_nodes != baseline_nodes:
            print(
                f"STRUCTURAL MISMATCH: {rel_path} ({result_nodes} vs {baseline_nodes} nodes)"
            )
            structural_mismatches += 1
            failed += 1
            continue

        # Numerical check - get test inputs
        benchmark_file = (
            benchmarks_path / rel_path.parent / Path("onnx") / rel_path.name
        )
        test_inputs = None

        if use_vnnlib_inputs and os.path.exists(benchmark_file):
            test_inputs = load_vnnlib_inputs(str(benchmark_file))

        if test_inputs is None:
            raise RuntimeError
            test_inputs = _generate_random_inputs(result_model, num_samples)

        # Run both models and compare
        all_match = True
        for inputs in test_inputs:
            try:
                result_outputs = _run_onnx_model(str(result_file), inputs)
                baseline_outputs = _run_onnx_model(str(baseline_file), inputs)

                match, mismatches = _compare_outputs(result_outputs, baseline_outputs)
                if not match:
                    all_match = False
                    break
            except Exception as e:
                raise e
                print(f"ERROR: {rel_path} - {e}")
                all_match = False
                break

        if all_match:
            passed += 1
        else:
            print(f"NUMERICAL MISMATCH: {rel_path}")
            numerical_mismatches += 1
            failed += 1

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(results_onnx)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"  Structural mismatches: {structural_mismatches}")
    print(f"  Numerical mismatches: {numerical_mismatches}")
    print(f"Missing baselines: {missing}")
    print(
        f"Pass rate: {passed/(len(results_onnx)-missing)*100:.1f}%"
        if len(results_onnx) > missing
        else "N/A"
    )

    return {
        "total": len(results_onnx),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "structural_mismatches": structural_mismatches,
        "numerical_mismatches": numerical_mismatches,
    }


def verify_against_benchmarks(
    results_dir: str = "results/baselines",
    benchmarks_dir: str = "benchmarks",
    num_samples: int = 5,
) -> dict:
    """Verify optimized results against original benchmark models.

    Checks that optimized models produce same outputs as original models.

    :param results_dir: Directory containing optimized results
    :param benchmarks_dir: Directory containing original benchmark models
    :param num_samples: Number of random samples for testing
    :return: Dictionary with verification results
    """
    results_path = Path(results_dir)
    benchmarks_path = Path(benchmarks_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

    print(f"Verifying {results_dir}/ against original {benchmarks_dir}/")
    print("=" * 70)

    results_onnx = sorted(results_path.rglob("*.onnx"))

    passed = 0
    failed = 0
    missing = 0

    for result_file in results_onnx:
        rel_path = result_file.relative_to(results_path)
        benchmark_file = (
            benchmarks_path / rel_path.parent / Path("onnx") / rel_path.name
        )

        if not benchmark_file.exists():
            print(f"MISSING: {rel_path} (no original benchmark)")
            missing += 1
            continue

        # Load models
        result_model = onnx.load(str(result_file))

        # Generate test inputs
        test_inputs = load_vnnlib_inputs(str(benchmark_file))

        if test_inputs is None:
            original_model = onnx.load(str(benchmark_file))
            test_inputs = _generate_random_inputs(original_model, num_samples)

        # Run both models and compare
        all_match = True
        for inputs in test_inputs:
            try:
                result_outputs = _run_onnx_model(str(result_file), inputs)
                benchmark_outputs = _run_onnx_model(str(benchmark_file), inputs)

                match, mismatches = _compare_outputs(result_outputs, benchmark_outputs)
                if not match:
                    print(f"MISMATCH: {rel_path}")
                    for msg in mismatches[:3]:
                        print(f"  {msg}")
                    all_match = False
                    break
            except Exception as e:
                print(f"ERROR: {rel_path} - {e}")
                raise e
                all_match = False
                break

        if all_match:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(results_onnx)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Missing originals: {missing}")
    print(
        f"Pass rate: {passed/(len(results_onnx)-missing)*100:.1f}%"
        if len(results_onnx) > missing
        else "N/A"
    )

    return {
        "total": len(results_onnx),
        "passed": passed,
        "failed": failed,
        "missing": missing,
    }


def main() -> None:
    """Main entry point for script execution."""
    import sys

    if "--create" in sys.argv:
        optimize_all_models()
    elif "--save-baseline" in sys.argv:
        save_as_baseline()
    elif "--verify" in sys.argv:
        verify_against_baseline()
    elif "--verify-benchmarks" in sys.argv:
        verify_against_benchmarks()
    else:
        print("Usage:")
        print("  python test_baselines.py --create              # Optimize to results/")
        print(
            "  python test_baselines.py --save-baseline       # Save results/ as baselines/"
        )
        print(
            "  python test_baselines.py --verify              # Verify results/ vs baselines/"
        )
        print(
            "  python test_baselines.py --verify-benchmarks   # Verify results/ vs benchmarks/"
        )
        sys.exit(1)


if __name__ == "__main__":
    # optimize_all_models()
    # save_as_baseline()
    # verify_against_baseline()
    verify_against_benchmarks()
    # main()
