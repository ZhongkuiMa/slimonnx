"""Utility functions for SlimONNX testing.

Extracted helper functions used across multiple test files.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "find_benchmarks_folders",
    "find_onnx_folders",
    "find_all_onnx_files",
    "find_all_vnnlib_files",
    "get_benchmark_name",
    "load_onnx_model",
    "if_has_batch_dim",
    "check_shape_compatibility",
    "infer_shape",
    "load_vnnlib_inputs",
    "compare_onnx_outputs",
]

import os
from pathlib import Path

import numpy as np


def find_benchmarks_folders(base_dir):
    """Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    benchmark_dirs = []
    # Only consider first-level subdirectories
    for entry in os.listdir(base_dir):
        subdir = os.path.normpath(os.path.join(base_dir, entry))
        if os.path.isdir(subdir):
            benchmark_dirs.append(subdir)
    return benchmark_dirs


def find_onnx_folders(benchmark_dirs):
    """Find ONNX subdirectories in benchmark directories.

    :param benchmark_dirs: List of benchmark directory paths
    :return: List of ONNX subdirectory paths
    """
    onnx_dirs = []
    for bdir in benchmark_dirs:
        onnx_subdir = os.path.join(bdir, "onnx")
        if os.path.isdir(onnx_subdir):
            onnx_dirs.append(onnx_subdir)
    return onnx_dirs


def find_all_onnx_files(benchmark_dirs, num_limit: int = 20):
    """Find all ONNX files in benchmark directories.

    Works with both structures:
    - benchmarks/benchmark_name/*.onnx (new structure)
    - benchmarks/benchmark_name/onnx/*.onnx (old structure)

    :param benchmark_dirs: List of benchmark directory paths
    :param num_limit: Maximum ONNX files per benchmark directory
    :return: List of ONNX file paths
    """
    onnx_files = []
    for bdir in benchmark_dirs:
        # Try new structure first (ONNX files directly in benchmark dir)
        i = 0
        for entry in os.listdir(bdir):
            if entry.endswith(".onnx"):
                onnx_path = os.path.normpath(os.path.join(bdir, entry))
                onnx_files.append(onnx_path)
                i += 1
                if i >= num_limit:
                    break

        # If no ONNX files found, try old structure (onnx subdirectory)
        if i == 0:
            onnx_subdir = os.path.join(bdir, "onnx")
            if os.path.isdir(onnx_subdir):
                for entry in os.listdir(onnx_subdir):
                    if entry.endswith(".onnx"):
                        onnx_path = os.path.normpath(os.path.join(onnx_subdir, entry))
                        onnx_files.append(onnx_path)
                        i += 1
                        if i >= num_limit:
                            break
    return onnx_files


def find_all_vnnlib_files(benchmark_dirs: list[str]) -> list[str]:
    """Find all VNNLib files in benchmark directories.

    :param benchmark_dirs: List of benchmark directory paths
    :return: List of VNNLib file paths
    """
    vnnlib_files = []
    for bdir in benchmark_dirs:
        # Look for vnnlib files directly in benchmark dir
        for entry in os.listdir(bdir):
            if entry.endswith(".vnnlib"):
                vnnlib_path = os.path.normpath(os.path.join(bdir, entry))
                vnnlib_files.append(vnnlib_path)

        # Also check vnnlib subdirectory if exists
        vnnlib_subdir = os.path.join(bdir, "vnnlib")
        if os.path.isdir(vnnlib_subdir):
            for entry in os.listdir(vnnlib_subdir):
                if entry.endswith(".vnnlib"):
                    vnnlib_path = os.path.normpath(os.path.join(vnnlib_subdir, entry))
                    vnnlib_files.append(vnnlib_path)

    return vnnlib_files


def get_benchmark_name(onnx_path: str, benchmarks_dir: str = "benchmarks") -> str:
    """Extract benchmark name from ONNX file path.

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Benchmark name (subdirectory name)
    """
    # Normalize path
    onnx_path = os.path.normpath(onnx_path)

    # Find benchmarks_dir in the path
    path_parts = onnx_path.split(os.sep)

    # Find the index of benchmarks_dir
    try:
        bench_idx = path_parts.index(benchmarks_dir)
        # The next part is the benchmark name
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    # Fallback: use the parent directory name
    return os.path.basename(os.path.dirname(onnx_path))


def load_onnx_model(onnx_path: str):
    """Load ONNX model and convert to version 21.

    :param onnx_path: Path to ONNX model file
    :return: ONNX ModelProto converted to version 21
    """
    import onnx

    model = onnx.load(onnx_path)
    model = onnx.version_converter.convert_version(model, target_version=21)
    return model


# Benchmarks/models without batch dimension
# Check by full path (can be benchmark name or model filename)
benchmark_without_batch_dim = [
    "cctsdb_yolo",
    "cgan",
    "pensieve_big_parallel.onnx",
    "pensieve_mid_parallel.onnx",
    "pensieve_small_parallel.onnx",
    "test_nano.onnx",
    "test_small.onnx",
    "test_tiny.onnx",
]


def if_has_batch_dim(onnx_path: str):
    """Determine if model has batch dimension by checking full path.

    Checks both benchmark name and model filename in the path.

    :param onnx_path: Path to ONNX model file
    :return: True if model has batch dimension, False otherwise
    """
    return all(bname not in onnx_path for bname in benchmark_without_batch_dim)


def check_shape_compatibility(inferred_shape, expected_shape) -> bool:
    """Check if inferred shape is compatible with expected shape.

    Allows scalar [] to match [1].

    :param inferred_shape: Shape inferred by shape inference
    :param expected_shape: Expected shape from model metadata
    :return: True if shapes are compatible, False otherwise
    """
    if inferred_shape == expected_shape:
        return True
    if inferred_shape == [] and expected_shape == [1]:
        return True
    return False


def infer_shape(model, has_batch_dim: bool = True, verbose: bool = False):
    """Run shape inference on model and validate against expected I/O shapes.

    :param model: ONNX ModelProto
    :param has_batch_dim: Whether model has batch dimension
    :param verbose: Whether to print verbose output during inference
    :return: Dictionary mapping tensor names to inferred shapes
    :raises ValueError: If inferred shapes don't match expected shapes
    """
    from shapeonnx import infer_onnx_shape, extract_io_shapes
    from shapeonnx.shapeonnx.utils import (
        get_initializers,
        get_input_nodes,
        get_output_nodes,
        convert_constant_to_initializer,
    )

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)
    nodes = list(model.graph.node)
    nodes = convert_constant_to_initializer(nodes, initializers)

    data_shapes = infer_onnx_shape(
        input_nodes, output_nodes, nodes, initializers, has_batch_dim, verbose
    )

    expected_input_shapes = extract_io_shapes(input_nodes, has_batch_dim)
    expected_output_shapes = extract_io_shapes(output_nodes, has_batch_dim)

    # Validate input shapes
    for input_node in input_nodes:
        input_name = input_node.name
        shape = data_shapes[input_name]
        expected_shape = expected_input_shapes[input_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Input shape mismatch for '{input_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

    # Validate output shapes
    for output_name in output_nodes:
        output_name = output_name.name
        shape = data_shapes[output_name]
        expected_shape = expected_output_shapes[output_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Output shape mismatch for '{output_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

    return data_shapes


def load_vnnlib_inputs(
    onnx_path: str,
    benchmarks_dir: str = "benchmarks",
) -> list[dict] | None:
    """Load test inputs from torchvnnlib .pth files.

    Reads .pth files created by torchvnnlib containing input bounds as tensors.
    Generates test inputs using the midpoint of bounds.

    Directory structure expected:
    benchmarks/benchmark_name/torchvnnlib/vnnlib_name/or_group_0/sub_prop_*.pth

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: List of input dictionaries, or None if not found
    """
    import onnx
    import torch

    # Find VNNLib name from instances.csv
    benchmark_dir = Path(onnx_path).parent.parent  # Go up from onnx/ to benchmark/
    torchvnnlib_dir = benchmark_dir / "torchvnnlib"

    if not torchvnnlib_dir.exists():
        return None
    instances_csv = benchmark_dir / "instances.csv"

    if not instances_csv.exists():
        return None

    # Parse instances.csv to find VNNLib file for this ONNX model
    onnx_rel_path = os.path.relpath(onnx_path, benchmark_dir)
    vnnlib_name = None

    try:
        with open(instances_csv) as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    model_path = parts[0].strip()
                    if model_path == onnx_rel_path:
                        vnnlib_path = parts[1].strip()
                        vnnlib_name = Path(vnnlib_path).stem
                        break
    except Exception:
        return None

    if vnnlib_name is None:
        return None

    # Look for .pth files in or_group_0 (taking first OR group only)
    vnnlib_dir = torchvnnlib_dir / vnnlib_name / "or_group_0"

    if not vnnlib_dir.exists():
        return None

    # Find all sub_prop_*.pth files
    pth_files = sorted(vnnlib_dir.glob("sub_prop_*.pth"))

    if not pth_files:
        return None

    # Load ONNX model to get input name
    model = onnx.load(onnx_path)
    input_names = [
        inp.name
        for inp in model.graph.input
        if not any(init.name == inp.name for init in model.graph.initializer)
    ]

    if len(input_names) == 0:
        return None

    input_name = input_names[0]

    # Load inputs from all .pth files
    inputs_list = []

    for pth_file in pth_files:
        try:
            data = torch.load(pth_file, weights_only=True)
            input_bounds = data["input"]  # Shape: (n_inputs, 2)

            # Use midpoint of bounds as test input
            lower = input_bounds[:, 0]
            upper = input_bounds[:, 1]
            midpoint = (lower + upper) / 2

            # Convert to numpy and reshape for ONNX Runtime
            arr = midpoint.numpy().astype(np.float32)
            inputs_list.append(
                {input_name: arr.reshape(1, -1) if arr.ndim == 1 else arr}
            )
        except Exception:
            continue

    return inputs_list if inputs_list else None


def compare_onnx_outputs(
    outputs1: dict,
    outputs2: dict,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Compare outputs from two ONNX models.

    :param outputs1: First model outputs (dict of arrays)
    :param outputs2: Second model outputs (dict of arrays)
    :param rtol: Relative tolerance for comparison
    :param atol: Absolute tolerance for comparison
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
            mean_diff = np.mean(np.abs(out1 - out2))
            mismatches.append(
                f"  {key}: max diff {max_diff:.2e}, mean diff {mean_diff:.2e}"
            )

    return len(mismatches) == 0, mismatches
