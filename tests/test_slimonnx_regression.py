"""Regression testing for SlimONNX against baselines.

This module provides regression tests that:
1. Create baselines with optimization statistics
2. Verify current results match stored baselines
3. Track I/O consistency over time
"""

__docformat__ = "restructuredtext"

import json
from pathlib import Path
from typing import Any, cast

import pytest

from slimonnx.tests.benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
)
from slimonnx.tests.test_slimonnx import (
    compare_outputs,
    optimize_model_with_slimonnx,
    run_onnx_model,
)
from slimonnx.tests.utils import load_test_inputs


def get_benchmark_models():
    """Collect all models from vnncomp2024 benchmarks."""
    test_dir = Path(__file__).parent
    benchmarks_dir = test_dir / "vnncomp2024_benchmarks"
    benchmarks = find_benchmarks(str(benchmarks_dir))
    models = find_models(benchmarks, max_per_benchmark=20)
    return [str(m) for m in models]


def get_baseline_path(model_path: str, baselines_dir: str | Path) -> str:
    """Get baseline JSON path for a model.

    :param model_path: Path to ONNX model file
    :param baselines_dir: Root directory for baseline files
    :return: Path to baseline JSON file
    """
    model_path_obj = Path(model_path)
    benchmark_name = get_model_benchmark_name(model_path_obj)
    return str(Path(baselines_dir) / benchmark_name / f"{model_path_obj.stem}.json")


def load_baseline(baseline_path: str) -> dict[str, Any] | None:
    """Load baseline data from JSON file.

    :param baseline_path: Path to baseline JSON file
    :return: Baseline data dictionary, or None if file not found
    """
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        return None

    with baseline_file.open() as f:
        return cast(dict[str, Any], json.load(f))


def compare_optimization_stats(current: dict, baseline: dict) -> list[str]:
    """Compare optimization statistics.

    :param current: Current optimization statistics
    :param baseline: Baseline optimization statistics
    :return: List of differences found
    """
    differences = []

    # Compare node counts
    if current["optimized_nodes"] != baseline["optimized_nodes"]:
        differences.append(
            f"Optimized nodes changed: {baseline['optimized_nodes']} -> {current['optimized_nodes']}"
        )

    # Compare reduction percentage (allow small variance)
    baseline_pct = baseline.get("reduction_pct", 0.0)
    current_pct = current.get("reduction_pct", 0.0)
    if abs(current_pct - baseline_pct) > 1.0:  # Allow 1% variance
        differences.append(
            f"Reduction percentage changed: {baseline_pct:.1f}% -> {current_pct:.1f}%"
        )

    return differences


@pytest.fixture(scope="session")
def baselines_dir():
    """Baseline directory for storing regression test data."""
    test_dir = Path(__file__).parent
    return test_dir / "baselines"


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_create_baseline(model_path, baselines_dir):
    """Create or update baseline for each model.

    This test creates baselines with optimization stats and I/O validation.

    :param model_path: Path to ONNX model file
    :param baselines_dir: Directory to store baselines
    """
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

    # Create baseline data
    baseline_data = {
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

    # Save baseline
    baseline_file = Path(get_baseline_path(model_path, baselines_dir))
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    with baseline_file.open("w") as f:
        json.dump(baseline_data, f, indent=2)

    # Verify baseline was created
    assert baseline_file.exists()


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_verify_baseline(model_path, baselines_dir):
    """Verify current optimization matches stored baseline.

    :param model_path: Path to ONNX model file
    :param baselines_dir: Directory containing baselines
    """
    baseline_path = get_baseline_path(model_path, baselines_dir)
    baseline_data = load_baseline(baseline_path)

    # Skip if no baseline exists yet
    if baseline_data is None:
        pytest.skip(f"No baseline for {Path(model_path).name}")

    baseline_stats = baseline_data["optimization_stats"]

    # Load test inputs
    try:
        test_inputs = load_test_inputs(model_path)
    except FileNotFoundError:
        pytest.skip("No test data available")

    if not test_inputs:
        pytest.skip("No test inputs found")

    # Optimize with SlimONNX
    optimized_path, current_stats = optimize_model_with_slimonnx(model_path)

    # Compare with baseline
    differences = compare_optimization_stats(current_stats, baseline_stats)

    if differences:
        pytest.fail(
            "Optimization differs from baseline:\n" + "\n".join(f"  - {d}" for d in differences)
        )

    # If we get here, stats match baseline
    assert current_stats["optimized_nodes"] == baseline_stats["optimized_nodes"]


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_io_consistency(model_path):
    """Verify optimized model produces consistent outputs.

    :param model_path: Path to ONNX model file
    """
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
    optimized_path, _ = optimize_model_with_slimonnx(model_path)

    # Get optimized outputs
    optimized_outputs = run_onnx_model(optimized_path, inputs)

    # Compare outputs
    max_diff, mean_diff = compare_outputs(original_outputs, optimized_outputs)

    # Validate I/O consistency
    assert max_diff < 1e-5, f"Max difference {max_diff} exceeds tolerance"
    assert mean_diff < 1e-6, f"Mean difference {mean_diff} exceeds tolerance"
