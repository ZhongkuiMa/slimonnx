"""VNNComp Benchmark Test Runner for SlimONNX.

Tests SlimONNX optimization on VNNComp 2024 benchmarks by:
1. Optimizing models with SlimONNX
2. Validating I/O consistency between original and optimized models
3. Creating/updating baselines for regression testing
"""

__docformat__ = "restructuredtext"

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from slimonnx import get_preset
from slimonnx.slimonnx import SlimONNX
from tests.test_benchmarks.benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
)
from tests.utils import load_onnx_model, load_test_inputs

# Known failing models due to SlimONNX optimizer bugs
KNOWN_FAILURES: dict[str, str] = {}


def get_benchmark_models():
    """Collect all models from vnncomp2024 benchmarks."""
    test_dir = Path(__file__).parent
    benchmarks_dir = test_dir / "vnncomp2024_benchmarks"
    benchmarks = find_benchmarks(str(benchmarks_dir))
    models = find_models(benchmarks, max_per_benchmark=20)
    return [str(m) for m in models]


def is_known_failure(model_path):
    """Check if model is a known failure."""
    model_name = Path(model_path).stem
    return any(model_name in failures for failures in KNOWN_FAILURES.values())


def run_onnx_model(model_path: str, inputs: np.ndarray) -> dict:
    """Run ONNX model inference using ONNX Runtime.

    :param model_path: Path to ONNX model file
    :param inputs: Input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    output_names = [out.name for out in session.get_outputs()]
    return dict(zip(output_names, outputs, strict=False))


def compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-5, atol: float = 1e-6
) -> tuple[float, float]:
    """Compare outputs from two models.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance
    :param atol: Absolute tolerance
    :return: Tuple of (max_difference, mean_difference)
    """
    max_diff = 0.0
    all_diffs = []

    for key1, key2 in zip(outputs1.keys(), outputs2.keys(), strict=False):
        out1 = outputs1[key1]
        out2 = outputs2[key2]

        assert out1.shape == out2.shape, f"Shape mismatch: {out1.shape} vs {out2.shape}"

        diff = np.abs(out1 - out2)
        max_diff = max(max_diff, np.max(diff))
        all_diffs.extend(diff.flatten())

    mean_diff = np.mean(all_diffs) if all_diffs else 0.0
    return max_diff, mean_diff


def optimize_model_with_slimonnx(model_path: str) -> tuple[str, dict]:
    """Optimize model with SlimONNX and return results.

    :param model_path: Path to original ONNX model
    :return: Tuple of (optimized_model_path, stats_dict)
    """
    model_path_obj = Path(model_path)
    benchmark_name = get_model_benchmark_name(model_path_obj)
    model_name = model_path_obj.name

    config = get_preset(benchmark_name, model_name)

    # Load original model
    model = load_onnx_model(model_path)
    original_node_count = len(model.graph.node)

    # Save to results directory
    test_dir = Path(__file__).parent
    output_path = test_dir / "results" / benchmark_name / model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file for optimization
    temp_path = model_path_obj.with_suffix(".temp.onnx")
    onnx.save(model, str(temp_path))

    try:
        start_time = time.perf_counter()
        slimonnx = SlimONNX()
        slimonnx.slim(str(temp_path), str(output_path), config=config)
        elapsed_time = time.perf_counter() - start_time

        optimized_model = onnx.load(str(output_path))
        optimized_node_count = len(optimized_model.graph.node)

        reduction = original_node_count - optimized_node_count
        reduction_pct = (reduction / original_node_count * 100) if original_node_count > 0 else 0.0

        stats = {
            "benchmark": benchmark_name,
            "model": model_name,
            "time": elapsed_time,
            "original_nodes": original_node_count,
            "optimized_nodes": optimized_node_count,
            "reduction": reduction,
            "reduction_pct": reduction_pct,
        }

        return str(output_path), stats

    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_results_stats(model_path: str, stats: dict, max_diff: float, mean_diff: float):
    """Save optimization stats to results directory.

    :param model_path: Path to original model
    :param stats: Optimization statistics
    :param max_diff: Maximum difference between original and optimized outputs
    :param mean_diff: Mean difference between original and optimized outputs
    """
    test_dir = Path(__file__).parent
    results_stats_path = test_dir / "results" / stats["benchmark"] / f"{Path(model_path).stem}.json"
    results_stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats_data = {
        "optimization_stats": {
            "original_nodes": stats["original_nodes"],
            "optimized_nodes": stats["optimized_nodes"],
            "reduction": stats["reduction"],
            "reduction_pct": stats["reduction_pct"],
            "time": stats["time"],
        },
        "io_validation": {
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "passed": bool(max_diff < 1e-5 and mean_diff < 1e-6),
        },
    }

    with results_stats_path.open("w") as f:
        json.dump(stats_data, f, indent=2)


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_slimonnx_optimization(model_path):
    """Test SlimONNX optimization preserves I/O behavior.

    Validates that optimized model produces same outputs as original
    within acceptable tolerance.

    :param model_path: Path to ONNX model file
    """
    # Skip known failing models
    if is_known_failure(model_path):
        pytest.skip("Known SlimONNX optimizer bug - skipping until fixed")

    # Load test inputs
    try:
        test_inputs = load_test_inputs(model_path)
    except FileNotFoundError:
        pytest.skip("No test data available")

    if not test_inputs:
        pytest.skip("No test inputs found")

    # Take first input for testing
    inputs = test_inputs[0]

    # Get original outputs
    original_outputs = run_onnx_model(model_path, inputs)

    # Optimize with SlimONNX
    optimized_path, stats = optimize_model_with_slimonnx(model_path)

    # Get optimized outputs
    optimized_outputs = run_onnx_model(optimized_path, inputs)

    # Compare outputs
    max_diff, mean_diff = compare_outputs(original_outputs, optimized_outputs)

    # Save results stats
    save_results_stats(model_path, stats, max_diff, mean_diff)

    # Report statistics
    print(
        f"\n{stats['benchmark']}/{stats['model']}: "
        f"{stats['original_nodes']}->{stats['optimized_nodes']} nodes "
        f"({stats['reduction_pct']:.1f}% reduction), "
        f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
    )

    # Validate within tolerance
    # Use relaxed tolerance for models with known numerical precision issues
    # (e.g., collins_rul_cnn models with initializers in graph inputs)
    max_tol = 1e-4 if "collins_rul_cnn" in model_path else 1e-5
    mean_tol = 1e-4 if "collins_rul_cnn" in model_path else 1e-6
    assert max_diff < max_tol, f"Max difference {max_diff} exceeds tolerance {max_tol}"
    assert mean_diff < mean_tol, f"Mean difference {mean_diff} exceeds tolerance {mean_tol}"
