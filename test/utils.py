"""Utility functions for SlimONNX testing."""

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
    "load_test_data_from_file",
]

import os
from pathlib import Path

import numpy as np


def find_benchmarks_folders(base_dir: str) -> list[str]:
    """Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    benchmark_dirs = []
    for entry in os.listdir(base_dir):
        subdir = os.path.normpath(os.path.join(base_dir, entry))
        if os.path.isdir(subdir):
            benchmark_dirs.append(subdir)
    return benchmark_dirs


def find_onnx_folders(benchmark_dirs: list[str]) -> list[str]:
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


def find_all_onnx_files(benchmark_dirs: list[str], num_limit: int = 20) -> list[str]:
    """Find all ONNX files in benchmark directories.

    Works with multiple structures:
    - Uses instances.csv if available to find ONNX files in subdirectories
    - benchmarks/benchmark_name/*.onnx (direct structure)
    - benchmarks/benchmark_name/onnx/*.onnx (onnx subdirectory)

    :param benchmark_dirs: List of benchmark directory paths
    :param num_limit: Maximum ONNX files per benchmark directory
    :return: List of ONNX file paths
    """
    onnx_files = []
    for bdir in benchmark_dirs:
        i = 0
        found_files = set()

        instances_csv = os.path.join(bdir, "instances.csv")
        if os.path.exists(instances_csv):
            try:
                with open(instances_csv) as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        parts = line.strip().split(",")
                        if parts:
                            model_path = parts[0]
                            onnx_path = os.path.normpath(os.path.join(bdir, model_path))
                            if onnx_path not in found_files and os.path.exists(
                                onnx_path
                            ):
                                found_files.add(onnx_path)
                                onnx_files.append(onnx_path)
                                i += 1
                                if i >= num_limit:
                                    break
            except Exception:
                pass

        if i == 0:
            for entry in os.listdir(bdir):
                if entry.endswith(".onnx"):
                    onnx_path = os.path.normpath(os.path.join(bdir, entry))
                    onnx_files.append(onnx_path)
                    i += 1
                    if i >= num_limit:
                        break

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
        for entry in os.listdir(bdir):
            if entry.endswith(".vnnlib"):
                vnnlib_path = os.path.normpath(os.path.join(bdir, entry))
                vnnlib_files.append(vnnlib_path)

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
    onnx_path = os.path.normpath(onnx_path)
    path_parts = onnx_path.split(os.sep)

    try:
        bench_idx = path_parts.index(benchmarks_dir)
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    return os.path.basename(os.path.dirname(onnx_path))


def load_onnx_model(onnx_path: str):
    """Load ONNX model and convert to version 21.

    :param onnx_path: Path to ONNX model file
    :return: ONNX ModelProto converted to version 21
    """
    import onnx
    from slimonnx.slimonnx.preprocess.version_converter import convert_model_version

    model = onnx.load(onnx_path)
    model = convert_model_version(model, target_opset=21, warn_on_diff=False)
    return model


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


def if_has_batch_dim(onnx_path: str) -> bool:
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

    for input_node in input_nodes:
        input_name = input_node.name
        shape = data_shapes[input_name]
        expected_shape = expected_input_shapes[input_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Input shape mismatch for '{input_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

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
) -> list[np.ndarray] | None:
    """Load test inputs from vnnlib .npy files.

    Reads .npy files containing input bounds as arrays.
    Generates test inputs using the midpoint of bounds.

    Directory structure expected:
    benchmarks/benchmark_name/vnnlib_data/vnnlib_name/or_group_0/sub_prop_*.npy

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: List of input dictionaries, or None if not found
    """
    import onnx

    benchmark_dir = Path(onnx_path).parent.parent
    vnnlib_data_dir = benchmark_dir / "vnnlib_data"

    if not vnnlib_data_dir.exists():
        return None

    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return None

    onnx_rel_path = os.path.relpath(onnx_path, benchmark_dir)
    vnnlib_name = None

    try:
        with open(instances_csv) as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    model_path = parts[0].strip()
                    if Path(model_path).name == Path(onnx_rel_path).name:
                        vnnlib_path = parts[1].strip()
                        vnnlib_name = Path(vnnlib_path).stem
                        break
    except Exception as e:
        raise e
        return None

    if vnnlib_name is None:
        return None

    vnnlib_dir = vnnlib_data_dir / vnnlib_name / "or_group_0"
    if not vnnlib_dir.exists():
        return None

    npz_files = sorted(vnnlib_dir.glob("sub_prop_*.npz"))
    if not npz_files:
        return None

    model = onnx.load(onnx_path)
    input_names = [
        inp.name
        for inp in model.graph.input
        if not any(init.name == inp.name for init in model.graph.initializer)
    ]

    if len(input_names) == 0:
        return None

    if len(input_names) != 1:
        raise NotImplementedError(
            "Loading VNNLib inputs for models with multiple inputs is not supported."
        )

    input_name = input_names[0]

    input_dims = None
    for inp in model.graph.input:
        if inp.name == input_name:
            input_dims = inp.type.tensor_type.shape.dim
            break

    inputs_list = []

    for npz_file in npz_files:
        try:
            input_bounds = np.load(npz_file)["input"]
            if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
                continue

            lower = input_bounds[:, 0]
            upper = input_bounds[:, 1]
            midpoint = (lower + upper) / 2

            arr = midpoint.astype(np.float32)

            if input_dims is not None:
                target_shape = []
                remaining_size = len(arr)

                for d in input_dims:
                    if d.dim_value > 0:
                        target_shape.append(d.dim_value)
                        remaining_size //= d.dim_value

                if len(target_shape) < len(input_dims):
                    target_shape.insert(0, remaining_size)

                arr = arr.reshape(target_shape)
            inputs_list.append(arr)
        except Exception as e:
            raise e
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


def load_test_data_from_file(data_path: str) -> list[dict[str, np.ndarray]]:
    """Load test input-output data from .npy or .npz file.

    :param data_path: Path to test data file
    :return: List of input dictionaries
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    if data_path.endswith(".npy"):
        data = np.load(data_path)
        if len(data.shape) == 1:
            return [{"input": data}]
        else:
            return [{"input": data[i]} for i in range(data.shape[0])]

    elif data_path.endswith(".npz"):
        data = np.load(data_path)
        if "inputs" in data:
            inputs_data = data["inputs"]
            return [{"input": inputs_data[i]} for i in range(inputs_data.shape[0])]
        elif "X" in data:
            X = data["X"]
            return [{"input": X[i]} for i in range(X.shape[0])]
        else:
            key = list(data.keys())[0]
            arr = data[key]
            return [{"input": arr[i]} for i in range(arr.shape[0])]

    else:
        raise ValueError(f"Unsupported file format: {data_path}")
