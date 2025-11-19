"""Baseline management for SlimONNX regression testing.

This module provides functions to create and compare optimization baselines.
Each ONNX model has its own baseline: the optimized ONNX file stored in baselines/.

Usage::

    # Create/update baseline for one model
    create_baseline("path/to/model.onnx")

    # Compare one model against baseline
    compare_baseline("path/to/model.onnx")

    # Batch: Create baselines for all benchmarks
    create_all_baselines()
"""

__docformat__ = "restructuredtext"
__all__ = [
    "get_baseline_path",
    "get_optimization_config",
    "generate_random_inputs",
    "run_onnx_model",
    "compare_outputs",
    "create_baseline",
    "compare_baseline",
    "create_all_baselines",
    "verify_all_baselines",
    "extract_benchmark_io_data",
    "compare_with_benchmark_data",
]

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slimonnx import SlimONNX
from slim_kwargs import SLIM_KWARGS
from utils import (
    find_all_onnx_files,
    find_benchmarks_folders,
    get_benchmark_name,
    if_has_batch_dim,
    load_onnx_model,
    load_vnnlib_inputs,
)


def get_baseline_path(onnx_path: str, baselines_dir: str = "baselines") -> str:
    """Get baseline ONNX path for a source ONNX model.

    The baseline file is stored in a subdirectory matching the benchmark name.
    For example: benchmarks/acasxu_2023/model.onnx -> baselines/acasxu_2023/model.onnx

    :param onnx_path: Path to source ONNX model file
    :param baselines_dir: Root directory to store baseline files
    :return: Path to baseline ONNX file
    """
    # Extract benchmark name from path
    benchmark_name = get_benchmark_name(onnx_path)

    # Get model basename
    basename = os.path.basename(onnx_path)

    # Create path with benchmark subdirectory
    return os.path.join(baselines_dir, benchmark_name, basename)


def get_optimization_config(benchmark_name: str) -> dict:
    """Get optimization configuration for a benchmark.

    :param benchmark_name: Name of the benchmark
    :return: Dictionary of optimization kwargs
    """
    return dict(SLIM_KWARGS[benchmark_name])


def generate_random_inputs(model: onnx.ModelProto, num_samples: int = 5) -> list[dict]:
    """Generate random test inputs matching model input signature.

    :param model: ONNX ModelProto
    :param num_samples: Number of random input samples to generate
    :return: List of input dictionaries
    """
    inputs_list = []

    for _ in range(num_samples):
        inputs = {}
        for inp in model.graph.input:
            # Skip initializers
            if any(init.name == inp.name for init in model.graph.initializer):
                continue

            # Get shape
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    # Dynamic dimension, use 1
                    shape.append(1)
                else:
                    shape.append(1)

            # Get data type
            dtype_map = {
                onnx.TensorProto.FLOAT: np.float32,
                onnx.TensorProto.DOUBLE: np.float64,
                onnx.TensorProto.INT32: np.int32,
                onnx.TensorProto.INT64: np.int64,
            }
            dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)

            # Generate random input
            if dtype in [np.float32, np.float64]:
                inputs[inp.name] = np.random.randn(*shape).astype(dtype)
            else:
                inputs[inp.name] = np.random.randint(0, 10, size=shape, dtype=dtype)

        inputs_list.append(inputs)

    return inputs_list


def run_onnx_model(model_path: str, inputs: dict) -> dict:
    """Run ONNX model inference using ONNX Runtime.

    :param model_path: Path to ONNX model file
    :param inputs: Dictionary of input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)

    # Map outputs to names
    output_names = [out.name for out in session.get_outputs()]
    return {name: output for name, output in zip(output_names, outputs)}


def compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[bool, list[str]]:
    """Compare outputs from two models.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Tuple of (all_match, mismatch_messages)
    """
    if set(outputs1.keys()) != set(outputs2.keys()):
        return False, [
            f"Output keys mismatch: {set(outputs1.keys())} vs {set(outputs2.keys())}"
        ]

    mismatches = []
    for key in outputs1.keys():
        out1 = outputs1[key]
        out2 = outputs2[key]

        if out1.shape != out2.shape:
            mismatches.append(f"  {key}: shape {out1.shape} vs {out2.shape}")
            continue

        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(out1 - out2))
            mismatches.append(f"  {key}: max diff {max_diff:.2e}")

    return len(mismatches) == 0, mismatches


def create_baseline(
    onnx_path: str, baselines_dir: str = "baselines", verbose: bool = False
) -> str:
    """Create baseline for ONE ONNX model.

    Process:
    1. Load original model and convert to ONNX v22
    2. Get benchmark name and optimization config
    3. Run SlimONNX.slim() to optimize
    4. Save optimized model to baselines/

    :param onnx_path: Path to source ONNX model file
    :param baselines_dir: Root directory to store baseline files
    :param verbose: Whether to print verbose output
    :return: Path to created baseline file
    """
    # Get optimization config
    benchmark_name = get_benchmark_name(onnx_path)
    opt_config = get_optimization_config(benchmark_name)
    has_batch_dim = if_has_batch_dim(onnx_path)

    # Create temporary v22 file
    model = load_onnx_model(onnx_path)
    temp_v22_path = onnx_path.replace(".onnx", "_temp_v22.onnx")
    onnx.save(model, temp_v22_path)

    # Get baseline path
    baseline_path = get_baseline_path(onnx_path, baselines_dir)
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)

    # Run optimization
    try:
        slimonnx = SlimONNX(verbose=verbose)
        slimonnx.slim(
            temp_v22_path,
            baseline_path,
            has_batch_dim=has_batch_dim,
            **opt_config,
        )

        print(
            f"[{benchmark_name}] Created baseline: {os.path.basename(onnx_path)} ({len(opt_config)} optimizations)"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_v22_path):
            os.remove(temp_v22_path)

    return baseline_path


def compare_baseline(
    onnx_path: str,
    baselines_dir: str = "baselines",
    use_vnnlib_inputs: bool = True,
    verbose: bool = False,
    num_test_samples: int = 5,
) -> bool:
    """Compare ONE ONNX model against its baseline.

    Process:
    1. Load baseline optimized model
    2. Re-optimize source model with current code
    3. Load torchvnnlib-extracted inputs or generate random inputs
    4. Run both models through ONNX Runtime
    5. Compare outputs with tolerance

    :param onnx_path: Path to source ONNX model file
    :param baselines_dir: Root directory containing baseline files
    :param use_vnnlib_inputs: Whether to use torchvnnlib inputs if available
    :param verbose: Whether to print verbose output
    :param num_test_samples: Number of random input samples to test (fallback)
    :return: True if outputs match baseline, False otherwise
    """
    baseline_path = get_baseline_path(onnx_path, baselines_dir)

    if not os.path.exists(baseline_path):
        print(f"No baseline: {os.path.basename(onnx_path)}")
        return False

    # Get optimization config
    benchmark_name = get_benchmark_name(onnx_path)
    opt_config = get_optimization_config(benchmark_name)
    has_batch_dim = if_has_batch_dim(onnx_path)

    # Create temporary v22 file
    model = load_onnx_model(onnx_path)
    temp_v22_path = onnx_path.replace(".onnx", "_temp_v22.onnx")
    onnx.save(model, temp_v22_path)

    # Re-optimize with current code
    temp_optimized_path = onnx_path.replace(".onnx", "_temp_optimized.onnx")

    try:
        slimonnx = SlimONNX(verbose=verbose)
        slimonnx.slim(
            temp_v22_path,
            temp_optimized_path,
            has_batch_dim=has_batch_dim,
            **opt_config,
        )

        # Load both models for comparison
        baseline_model = onnx.load(baseline_path)
        current_model = onnx.load(temp_optimized_path)

        # Quick structural check
        baseline_node_count = len(baseline_model.graph.node)
        current_node_count = len(current_model.graph.node)

        if baseline_node_count != current_node_count:
            print(
                f"MISMATCH: {os.path.basename(onnx_path)} - node count {baseline_node_count} vs {current_node_count}"
            )
            return False

        # Get test inputs (try torchvnnlib first, fallback to random)
        test_inputs = None
        input_source = "random"
        if use_vnnlib_inputs:
            test_inputs = load_vnnlib_inputs(onnx_path)
            if test_inputs:
                input_source = "torchvnnlib"

        if test_inputs is None:
            test_inputs = generate_random_inputs(model, num_samples=num_test_samples)

        # Run both models and compare outputs
        all_match = True
        for i, inputs in enumerate(test_inputs):
            try:
                baseline_outputs = run_onnx_model(baseline_path, inputs)
                current_outputs = run_onnx_model(temp_optimized_path, inputs)

                match, mismatches = compare_outputs(baseline_outputs, current_outputs)

                if not match:
                    if i == 0:  # Only print on first mismatch
                        print(f"MISMATCH: {os.path.basename(onnx_path)}")
                    print(f"  Sample {i+1}: Output differences:")
                    for msg in mismatches:
                        print(f"    {msg}")
                    all_match = False
            except Exception as e:
                print(f"ERROR running sample {i+1}: {e}")
                all_match = False

        if all_match:
            print(
                f"OK: {os.path.basename(onnx_path)} ({input_source} inputs, {len(test_inputs)} samples)"
            )

        return all_match

    finally:
        # Clean up temp files
        if os.path.exists(temp_v22_path):
            os.remove(temp_v22_path)
        if os.path.exists(temp_optimized_path):
            os.remove(temp_optimized_path)


def create_all_baselines(
    benchmark_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    max_per_benchmark: int = 20,
    verbose: bool = False,
):
    """Create baselines for all benchmark models.

    :param benchmark_dir: Root directory of benchmarks
    :param baselines_dir: Root directory to store baseline files
    :param max_per_benchmark: Maximum models per benchmark to process
    :param verbose: Whether to print verbose output during optimization
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Creating baselines for {len(onnx_files)} models")

    success = 0
    failed = []
    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        print(f"[{i}/{len(onnx_files)}] ", end="")
        try:
            create_baseline(onnx_path, baselines_dir, verbose=verbose)
            success += 1
        except Exception as e:
            print(f"Error: {e}")
            failed.append((onnx_path, str(e)))

    total_time = time.perf_counter() - start_time

    print(f"\nCompleted: {success}/{len(onnx_files)} success, {len(failed)} failed")
    if failed:
        print("Failed models:")
        for onnx_path, error in failed:
            print(f"  {os.path.basename(onnx_path)}: {error}")
    print(
        f"Total time: {total_time:.2f}s (avg {total_time/len(onnx_files):.2f}s/model)"
    )


def verify_all_baselines(
    benchmark_dir: str = "benchmarks",
    baselines_dir: str = "baselines",
    max_per_benchmark: int = 20,
    use_vnnlib_inputs: bool = True,
    verbose: bool = False,
    num_test_samples: int = 5,
):
    """Verify all models against their baselines.

    :param benchmark_dir: Root directory of benchmarks
    :param baselines_dir: Root directory containing baseline files
    :param max_per_benchmark: Maximum models per benchmark to verify
    :param use_vnnlib_inputs: Whether to use torchvnnlib inputs if available
    :param verbose: Whether to print verbose output during optimization
    :param num_test_samples: Number of random input samples to test per model
    """
    benchmark_dirs = find_benchmarks_folders(benchmark_dir)
    onnx_files = find_all_onnx_files(benchmark_dirs, num_limit=max_per_benchmark)
    print(f"Verifying {len(onnx_files)} models")

    passed = 0
    failed = []
    missing = []
    start_time = time.perf_counter()

    for i, onnx_path in enumerate(onnx_files, 1):
        print(f"[{i}/{len(onnx_files)}] ", end="")
        try:
            baseline_path = get_baseline_path(onnx_path, baselines_dir)
            if not os.path.exists(baseline_path):
                print(f"Skip {os.path.basename(onnx_path)} - no baseline")
                missing.append(onnx_path)
                continue

            if compare_baseline(
                onnx_path,
                baselines_dir,
                use_vnnlib_inputs,
                verbose=verbose,
                num_test_samples=num_test_samples,
            ):
                passed += 1
            else:
                failed.append(onnx_path)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(onnx_path)

    total_time = time.perf_counter() - start_time
    tested = len(onnx_files) - len(missing)

    print(f"\nTested: {tested}/{len(onnx_files)}")
    print(f"Passed: {passed}/{tested}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"  {os.path.basename(f)}")
    if missing:
        print(f"Missing baselines: {len(missing)}")
    print(f"Total time: {total_time:.2f}s")

    return len(failed) == 0


def extract_benchmark_io_data(
    onnx_path: str,
    benchmark_dir: str,
    num_samples: int = 5,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """Extract input-output pairs from benchmark data.

    :param onnx_path: Path to ONNX model in benchmark
    :param benchmark_dir: Root benchmark directory
    :param num_samples: Number of I/O samples to extract
    :return: Tuple of (inputs, expected_outputs)
    """
    from pathlib import Path

    # Try to find VNNLib file with same base name
    model_name = Path(onnx_path).stem
    vnnlib_candidates = list(Path(benchmark_dir).rglob(f"{model_name}*.vnnlib"))

    if not vnnlib_candidates:
        # Try finding any vnnlib file in the benchmark
        vnnlib_candidates = list(Path(benchmark_dir).rglob("*.vnnlib"))

    # Try to extract bounds from VNNLib
    input_bounds = None
    if vnnlib_candidates:
        try:
            # Try to parse first VNNLib file for input bounds
            from utils import parse_vnnlib_input_bounds

            input_bounds = parse_vnnlib_input_bounds(str(vnnlib_candidates[0]))
        except Exception:
            pass

    # Load original model
    model = onnx.load(onnx_path)

    # Generate inputs
    if input_bounds is not None:
        # Use bounds from VNNLib
        from slimonnx.model_validate import generate_inputs_from_bounds

        input_shape = tuple(
            d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
        )
        inputs = generate_inputs_from_bounds(input_bounds, input_shape, num_samples)
    else:
        # Generate random inputs
        inputs = generate_random_inputs(model, num_samples)

    # Run original model to get expected outputs
    expected_outputs = []
    for inp in inputs:
        outputs = run_onnx_model(onnx_path, inp)
        expected_outputs.append(outputs)

    return inputs, expected_outputs


def compare_with_benchmark_data(
    optimized_model_path: str,
    benchmark_onnx_path: str,
    benchmark_dir: str,
    num_samples: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict:
    """Compare optimized model outputs with benchmark ground truth.

    :param optimized_model_path: Path to optimized model (from baselines/)
    :param benchmark_onnx_path: Path to original model (from benchmarks/)
    :param benchmark_dir: Root directory of benchmark
    :param num_samples: Number of test samples
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Comparison report
    """
    # Extract I/O data from benchmark
    test_inputs, expected_outputs = extract_benchmark_io_data(
        benchmark_onnx_path, benchmark_dir, num_samples
    )

    # Run optimized model
    actual_outputs = []
    for inp in test_inputs:
        outputs = run_onnx_model(optimized_model_path, inp)
        actual_outputs.append(outputs)

    # Compare outputs
    passed = 0
    failed = 0
    max_diff = 0.0
    mismatches = []

    for i, (expected, actual) in enumerate(zip(expected_outputs, actual_outputs)):
        match, diffs = compare_outputs(expected, actual, rtol, atol)

        if match:
            passed += 1
        else:
            failed += 1
            for diff in diffs:
                if "max_diff" in diff:
                    max_diff = max(max_diff, diff["max_diff"])
            mismatches.append(f"Test {i}: {[d['message'] for d in diffs]}")

    return {
        "all_match": failed == 0,
        "num_tests": len(test_inputs),
        "passed": passed,
        "failed": failed,
        "max_diff": max_diff,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    # Example 1: Create baseline for one model
    # create_baseline("benchmarks/acasxu_2023/ACASXU_run2a_1_1_batch_2000.onnx")

    # Example 2: Compare one model against baseline
    # compare_baseline("benchmarks/acasxu_2023/ACASXU_run2a_1_1_batch_2000.onnx")

    # Example 3: Create baselines for all benchmarks
    # create_all_baselines(verbose=False)

    # Example 4: Verify all benchmarks against baselines
    verify_all_baselines(verbose=False, num_test_samples=5)
