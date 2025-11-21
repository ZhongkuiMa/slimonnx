"""Utility functions for SlimONNX testing."""

__docformat__ = "restructuredtext"
__all__ = [
    "find_benchmark_folders",
    "find_onnx_files_from_instances",
    "find_vnnlib_files_from_instances",
    "get_benchmark_name",
    "get_benchmark_dir",
    "load_onnx_model",
    "if_has_batch_dim",
    "check_shape_compatibility",
    "infer_shape",
    "load_test_inputs",
    "compare_onnx_outputs",
    "load_test_data_from_file",
    "BENCHMARKS_WITHOUT_BATCH_DIM",
]

from pathlib import Path

import numpy as np


def find_benchmark_folders(base_dir: str) -> list[str]:
    """Find all benchmark directories in base_dir.

    :param base_dir: Root directory containing benchmark subdirectories
    :return: List of benchmark directory paths
    """
    benchmark_dirs = []
    base_path = Path(base_dir)
    for entry in base_path.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            benchmark_dirs.append(str(entry))
    return benchmark_dirs


def find_onnx_files_from_instances(
    benchmark_dirs: list[str], num_limit: int = 20
) -> list[str]:
    """Find ONNX files from instances.csv in benchmark directories.

    Reads instances.csv to get ONNX file paths. All files are expected
    to be in their original structure as referenced in instances.csv.

    :param benchmark_dirs: List of benchmark directory paths
    :param num_limit: Maximum ONNX files per benchmark directory
    :return: List of ONNX file paths
    """
    onnx_files = []
    for bdir in benchmark_dirs:
        bdir_path = Path(bdir)
        instances_csv = bdir_path / "instances.csv"
        if not instances_csv.exists():
            continue

        found_files = set()
        try:
            with open(instances_csv) as file_handle:
                for line in file_handle.readlines()[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if parts:
                        model_path = parts[0].strip()
                        onnx_path = str((bdir_path / model_path).resolve())
                        if onnx_path not in found_files and Path(onnx_path).exists():
                            found_files.add(onnx_path)
                            onnx_files.append(onnx_path)
                            if len(found_files) >= num_limit:
                                break
        except OSError as error:
            print(f"Error reading instances.csv in {bdir}: {error}")
            continue

    return onnx_files


def find_vnnlib_files_from_instances(benchmark_dirs: list[str]) -> list[str]:
    """Find VNNLib files from instances.csv in benchmark directories.

    Reads instances.csv to get VNNLib file paths. All files are expected
    to be in their original structure as referenced in instances.csv.

    :param benchmark_dirs: List of benchmark directory paths
    :return: List of VNNLib file paths
    """
    vnnlib_files = []
    for bdir in benchmark_dirs:
        bdir_path = Path(bdir)
        instances_csv = bdir_path / "instances.csv"
        if not instances_csv.exists():
            continue

        found_files = set()
        try:
            with open(instances_csv) as file_handle:
                for line in file_handle.readlines()[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 2:
                        vnnlib_path_rel = parts[1].strip()
                        vnnlib_path = str((bdir_path / vnnlib_path_rel).resolve())
                        if (
                            vnnlib_path not in found_files
                            and Path(vnnlib_path).exists()
                        ):
                            found_files.add(vnnlib_path)
                            vnnlib_files.append(vnnlib_path)
        except OSError as error:
            print(f"Error reading instances.csv in {bdir}: {error}")
            continue

    return vnnlib_files


def get_benchmark_name(onnx_path: str, benchmarks_dir: str = "benchmarks") -> str:
    """Extract benchmark name from ONNX file path.

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Benchmark name (subdirectory name)
    """
    path_obj = Path(onnx_path).resolve()
    path_parts = path_obj.parts

    try:
        bench_idx = path_parts.index(benchmarks_dir)
        if bench_idx + 1 < len(path_parts):
            return path_parts[bench_idx + 1]
    except ValueError:
        pass

    return path_obj.parent.name


def get_benchmark_dir(onnx_path: str, benchmarks_dir: str = "benchmarks") -> Path:
    """Find the benchmark root directory for a given ONNX file.

    Searches upward from the ONNX file path to find the benchmark directory
    that contains instances.csv.

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: Path to benchmark directory
    :raises FileNotFoundError: If benchmark directory cannot be found
    """
    current = Path(onnx_path).parent

    # Search upward for directory containing instances.csv
    max_depth = 5
    for _ in range(max_depth):
        if (current / "instances.csv").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # Fallback: try to find by benchmarks_dir name
    current = Path(onnx_path)
    for parent in current.parents:
        if parent.name == benchmarks_dir and parent.parent.name != benchmarks_dir:
            # Found benchmarks/ root, get the benchmark subdirectory
            rel = Path(onnx_path).relative_to(parent)
            if rel.parts:
                benchmark_subdir = parent / rel.parts[0]
                if (benchmark_subdir / "instances.csv").exists():
                    return benchmark_subdir

    raise FileNotFoundError(
        f"Could not find benchmark directory with instances.csv for {onnx_path}"
    )


def load_onnx_model(onnx_path: str):  # type: ignore
    """Load ONNX model and convert to version 21.

    :param onnx_path: Path to ONNX model file
    :return: ONNX ModelProto converted to version 21
    """
    import onnx
    from slimonnx.slimonnx.preprocess.version_converter import convert_model_version

    model = onnx.load(onnx_path)
    model = convert_model_version(model, target_opset=21, warn_on_diff=False)
    return model


BENCHMARKS_WITHOUT_BATCH_DIM = (
    "cctsdb_yolo",
    "cgan",
    "pensieve_big_parallel.onnx",
    "pensieve_mid_parallel.onnx",
    "pensieve_small_parallel.onnx",
    "test_nano.onnx",
    "test_small.onnx",
    "test_tiny.onnx",
)


def if_has_batch_dim(onnx_path: str) -> bool:
    """Determine if model has batch dimension by checking full path.

    Checks both benchmark name and model filename in the path.

    :param onnx_path: Path to ONNX model file
    :return: True if model has batch dimension, False otherwise
    """
    return all(bname not in onnx_path for bname in BENCHMARKS_WITHOUT_BATCH_DIM)


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


def infer_shape(
    model, has_batch_dim: bool = True, verbose: bool = False
) -> dict[str, list[int]]:
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


def _load_precomputed_data(data_file: Path) -> list[np.ndarray]:
    """Load pre-computed test data from npz file.

    :param data_file: Path to npz data file
    :return: List of input arrays, empty if loading fails
    """
    if not data_file.exists():
        return []

    try:
        data = np.load(data_file, allow_pickle=True)
        inputs_list = []

        for vnnlib_name in data.files:
            vnnlib_data = data[vnnlib_name].item()
            if isinstance(vnnlib_data, dict):
                for bound_type in ["lower", "upper"]:
                    if bound_type in vnnlib_data:
                        bound_data = vnnlib_data[bound_type]
                        if "inputs" in bound_data:
                            inputs_list.extend(bound_data["inputs"])

        return inputs_list
    except (IOError, KeyError, ValueError):
        return []


def _get_vnnlib_names_from_csv(instances_csv: Path, onnx_rel_path: str) -> list[str]:
    """Extract VNNLib names for a model from instances.csv.

    :param instances_csv: Path to instances.csv file
    :param onnx_rel_path: Relative path to ONNX model
    :return: List of VNNLib file names
    :raises FileNotFoundError: If instances.csv not found or parsing fails
    """
    if not instances_csv.exists():
        raise FileNotFoundError(f"No instances.csv found at {instances_csv}")

    vnnlib_names = []
    try:
        with open(instances_csv) as file_handle:
            for line in file_handle.readlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    model_path = parts[0].strip().replace("\\", "/")
                    if model_path == onnx_rel_path:
                        vnnlib_path = parts[1].strip()
                        vnnlib_name = Path(vnnlib_path).stem
                        if vnnlib_name not in vnnlib_names:
                            vnnlib_names.append(vnnlib_name)
    except IOError as error:
        raise FileNotFoundError(f"Error reading instances.csv: {error}") from error

    return vnnlib_names


def _get_model_input_info(onnx_path: str) -> tuple[str, list]:
    """Get model input name and dimensions.

    :param onnx_path: Path to ONNX model
    :return: Tuple of (input_name, input_dims)
    :raises ValueError: If model has no inputs
    :raises NotImplementedError: If model has multiple inputs
    """
    import onnx

    model = onnx.load(onnx_path)
    input_names = [
        inp.name
        for inp in model.graph.input
        if not any(init.name == inp.name for init in model.graph.initializer)
    ]

    if len(input_names) == 0:
        raise ValueError(f"Model has no inputs: {onnx_path}")

    if len(input_names) != 1:
        raise NotImplementedError(
            f"Loading inputs for models with multiple inputs is not supported: {onnx_path}"
        )

    input_name = input_names[0]
    input_dims = None
    for inp in model.graph.input:
        if inp.name == input_name:
            input_dims = inp.type.tensor_type.shape.dim
            break

    return input_name, input_dims


def _reshape_array_to_input_dims(arr: np.ndarray, input_dims: list) -> np.ndarray:
    """Reshape flat array to match model input dimensions.

    :param arr: Flat input array
    :param input_dims: ONNX input dimensions
    :return: Reshaped array
    """
    if input_dims is None:
        return arr

    target_shape = []
    remaining_size = len(arr)

    for dim in input_dims:
        if dim.dim_value > 0:
            target_shape.append(dim.dim_value)
            remaining_size //= dim.dim_value

    if len(target_shape) < len(input_dims):
        target_shape.insert(0, remaining_size)

    return arr.reshape(target_shape)


def _load_from_vnnlib_data(
    vnnlib_data_dir: Path, vnnlib_names: list[str], input_dims: list
) -> list[np.ndarray]:
    """Load test inputs from vnnlib_data directory.

    :param vnnlib_data_dir: Path to vnnlib_data directory
    :param vnnlib_names: List of VNNLib file names
    :param input_dims: Model input dimensions for reshaping
    :return: List of input arrays
    """
    inputs_list = []

    for vnnlib_name in vnnlib_names:
        vnnlib_dir = vnnlib_data_dir / vnnlib_name / "or_group_0"
        if not vnnlib_dir.exists():
            continue

        npz_files = sorted(vnnlib_dir.glob("sub_prop_*.npz"))
        for npz_file in npz_files:
            try:
                input_bounds = np.load(npz_file)["input"]
                if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
                    continue

                lower = input_bounds[:, 0]
                upper = input_bounds[:, 1]
                midpoint = (lower + upper) / 2
                arr = midpoint.astype(np.float32)

                arr = _reshape_array_to_input_dims(arr, input_dims)
                inputs_list.append(arr)
            except (IOError, KeyError, ValueError, IndexError):
                continue

    return inputs_list


def load_test_inputs(
    onnx_path: str, benchmarks_dir: str = "benchmarks"
) -> list[np.ndarray]:
    """Load test inputs for an ONNX model.

    Tries in order:
    1. Pre-computed data from data/ directory (npz files from calculate_outputs)
    2. VNNLib-derived inputs from vnnlib_data/ directory

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :return: List of input arrays
    :raises FileNotFoundError: If no test data is available
    """
    try:
        benchmark_dir = get_benchmark_dir(onnx_path, benchmarks_dir)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Cannot load test inputs: {error}") from error

    model_name = Path(onnx_path).stem

    # Try pre-computed data first
    data_file = benchmark_dir / "data" / f"{model_name}.npz"
    precomputed_inputs = _load_precomputed_data(data_file)
    if precomputed_inputs:
        return precomputed_inputs

    # Fallback to vnnlib_data
    onnx_rel_path = str(Path(onnx_path).relative_to(benchmark_dir)).replace("\\", "/")
    instances_csv = benchmark_dir / "instances.csv"

    vnnlib_names = _get_vnnlib_names_from_csv(instances_csv, onnx_rel_path)
    if not vnnlib_names:
        raise FileNotFoundError(
            f"No VNNLib files found for {onnx_rel_path} in instances.csv"
        )

    vnnlib_data_dir = benchmark_dir / "vnnlib_data"
    if not vnnlib_data_dir.exists():
        raise FileNotFoundError(
            f"No vnnlib_data directory found in {benchmark_dir}. "
            f"Run extract_inputs() first."
        )

    input_name, input_dims = _get_model_input_info(onnx_path)
    inputs_list = _load_from_vnnlib_data(vnnlib_data_dir, vnnlib_names, input_dims)

    if not inputs_list:
        raise FileNotFoundError(
            f"No valid test inputs found for {onnx_path}. "
            f"Check that vnnlib_data exists and contains valid .npz files."
        )

    return inputs_list


def compare_onnx_outputs(  # type: ignore
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
    if not Path(data_path).exists():
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
