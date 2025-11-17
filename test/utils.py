"""Utility functions for ShapeONNX testing.

Extracted helper functions used across multiple test files.
"""

import os


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
