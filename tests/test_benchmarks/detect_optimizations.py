"""Optimization detection tool for benchmarks.

Analyzes ONNX models to detect which optimization patterns are present and
recommends OptimizationConfig settings.
"""

__docformat__ = "restructuredtext"
__all__ = ["detect_benchmark_optimizations", "generate_preset_config"]

from collections import defaultdict
from pathlib import Path

from slimonnx import OptimizationConfig
from slimonnx.slimonnx import SlimONNX
from tests.test_benchmarks.benchmark_utils import find_onnx_files_from_instances
from tests.utils import if_has_batch_dim


def detect_benchmark_optimizations(benchmark_dir: str, max_models: int = 5) -> dict:
    """Detect optimization patterns in a benchmark's ONNX models.

    :param benchmark_dir: Path to benchmark directory
    :param max_models: Maximum number of models to analyze
    :return: Dictionary with pattern counts and recommended optimizations
    """
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

    onnx_files = find_onnx_files_from_instances([str(benchmark_path)], num_limit=max_models)

    if not onnx_files:
        raise ValueError(f"No ONNX files found in {benchmark_dir}")

    print(f"Analyzing {len(onnx_files)} models from {benchmark_path.name}")
    print("=" * 70)

    slimonnx = SlimONNX()
    pattern_totals: defaultdict[str, int] = defaultdict(int)
    has_batch_dim_count = 0

    for i, onnx_path in enumerate(onnx_files, 1):
        basename = Path(onnx_path).name
        print(f"[{i}/{len(onnx_files)}] {basename}...", end=" ")

        has_batch = if_has_batch_dim(onnx_path)
        if has_batch:
            has_batch_dim_count += 1

        try:
            config = OptimizationConfig(has_batch_dim=has_batch)
            patterns = slimonnx.detect_patterns(onnx_path, config=config)

            pattern_count = sum(p["count"] for p in patterns.values())
            for pattern_name, data in patterns.items():
                if data["count"] > 0:
                    pattern_totals[pattern_name] += data["count"]

            print(f"OK ({pattern_count} patterns, batch_dim={has_batch})")

        except (ImportError, ValueError, AttributeError, RuntimeError) as error:
            print(f"FAILED: {error}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION DETECTION RESULTS")
    print("=" * 70)

    has_batch_dim = has_batch_dim_count > len(onnx_files) / 2

    print(f"Total models analyzed: {len(onnx_files)}")
    print(f"Models with batch dim: {has_batch_dim_count}/{len(onnx_files)}")
    print(f"Recommended has_batch_dim: {has_batch_dim}")
    print()

    if pattern_totals:
        print("Detected patterns:")
        sorted_patterns = sorted(pattern_totals.items(), key=lambda x: x[1], reverse=True)
        for pattern_name, count in sorted_patterns:
            print(f"  {pattern_name}: {count}")
    else:
        print("No optimization patterns detected")

    return {
        "benchmark_name": benchmark_path.name,
        "models_analyzed": len(onnx_files),
        "has_batch_dim": has_batch_dim,
        "patterns": dict(pattern_totals),
    }


def generate_preset_config(detection_result: dict) -> str:
    """Generate OptimizationConfig code based on detection results.

    :param detection_result: Result from detect_benchmark_optimizations
    :return: Python code string for OptimizationConfig
    """
    patterns = detection_result["patterns"]

    # Pattern to optimization flag mapping
    pattern_to_flag = {
        "matmul_add": "fuse_matmul_add=True",
        "conv_bn": "fuse_conv_bn=True",
        "bn_conv": "fuse_bn_conv=True",
        "convtransposed_bn": "fuse_convtransposed_bn=True",
        "bn_convtransposed": "fuse_bn_convtransposed=True",
        "gemm_reshape_bn": "fuse_gemm_reshape_bn=True",
        "bn_reshape_gemm": "fuse_bn_reshape_gemm=True",
        "bn_gemm": "fuse_bn_gemm=True",
        "transpose_bn_transpose": "fuse_transpose_bn_transpose=True",
        "gemm_gemm": "fuse_gemm_gemm=True",
        "conv_to_flatten_gemm": "simplify_conv_to_flatten_gemm=True",
    }

    optimizations = [flag for pattern, flag in pattern_to_flag.items() if pattern in patterns]

    if any("redundant" in p for p in patterns):
        optimizations.append("remove_redundant_operations=True")

    optimizations.append("constant_folding=True")

    config_lines = ["OptimizationConfig("]
    config_lines.extend(f"    {opt}," for opt in optimizations)
    config_lines.append(")")

    return "\n".join(config_lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detect_optimizations.py <benchmark_dir>")
        example_cmd = (
            "python detect_optimizations.py ../../tests/vnncomp2024/benchmarks/ml4acopf_2023"
        )
        print(f"Example: {example_cmd}")
        sys.exit(1)

    benchmark_dir = sys.argv[1]
    result = detect_benchmark_optimizations(benchmark_dir)

    print("\n" + "=" * 70)
    print("RECOMMENDED PRESET CONFIGURATION")
    print("=" * 70)
    print(f'"{result["benchmark_name"]}": {generate_preset_config(result)}')
