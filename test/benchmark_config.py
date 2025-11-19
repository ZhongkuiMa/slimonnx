__docformat__ = "restructuredtext"
__all__ = ["BENCHMARK_VERSIONS", "get_benchmark_version", "has_test_data"]

import os
import types

# Benchmark version configuration to preserve original IR/Opset versions
# Format: "benchmark_name": (target_ir_version, target_opset)
# None means keep original version
BENCHMARK_VERSIONS = types.MappingProxyType(
    {
        # VNNComp benchmarks - keep original versions to avoid compatibility issues
        "acasxu_2023": None,  # Keep original
        "cctsdb_yolo_2023": None,
        "cgan_2023": None,  # IR=4, Opset=9 - too old to downgrade from Opset 23
        "cifar100": None,  # IR=7, Opset=12 - works well
        "cora": None,
        "cora_2024": None,
        "dist_shift_2023": None,
        "linearizenn": None,
        "metaroom_2023": None,
        "safenlp": None,
        "safenlp_2024": None,
        "test": None,
        "tllverifybench_2023": None,
        "traffic_signs_recognition_2023": None,
        "vggnet16_2022": None,
        "vggnet16_2023": None,
        # Add more benchmarks as needed
        # "benchmark_name": (8, 17),  # Example: IR=8, Opset=17
    }
)


def get_benchmark_version(benchmark_name: str) -> tuple[int, int] | None:
    """Get target IR and Opset version for a benchmark.

    :param benchmark_name: Name of the benchmark
    :return: Tuple of (ir_version, opset_version) or None to keep original
    """
    return BENCHMARK_VERSIONS.get(benchmark_name, None)


def has_test_data(benchmark_name: str, base_dir: str = "benchmarks") -> bool:
    """Check if benchmark has test data in data folder.

    :param benchmark_name: Name of the benchmark
    :param base_dir: Base directory containing benchmarks
    :return: True if data folder exists with .pth or .npy files
    """
    data_dir = os.path.join(base_dir, benchmark_name, "data")
    if not os.path.exists(data_dir):
        return False

    # Check for .pth or .npy files
    for file in os.listdir(data_dir):
        if file.endswith((".pth", ".npy", ".npz")):
            return True

    return False


def get_test_data_path(
    onnx_path: str, benchmark_name: str, base_dir: str = "benchmarks"
) -> str | None:
    """Get path to test data file for a model.

    :param onnx_path: Path to ONNX model
    :param benchmark_name: Name of the benchmark
    :param base_dir: Base directory containing benchmarks
    :return: Path to test data file (.pth) or None if not found
    """
    model_name = os.path.basename(onnx_path).replace(".onnx", "")
    data_dir = os.path.join(base_dir, benchmark_name, "data")

    if not os.path.exists(data_dir):
        return None

    # Try .pth first, then .npy, then .npz
    for ext in [".pth", ".npy", ".npz"]:
        data_path = os.path.join(data_dir, model_name + ext)
        if os.path.exists(data_path):
            return data_path

    return None
