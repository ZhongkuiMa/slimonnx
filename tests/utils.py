"""Test-specific utility functions for SlimONNX."""

__docformat__ = "restructuredtext"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "check_shape_compatibility",
    "if_has_batch_dim",
    "infer_shape",
    "load_onnx_model",
    "load_test_inputs",
]

from pathlib import Path
from typing import Any

import numpy as np
import onnx

from tests.test_benchmarks.benchmark_utils import get_benchmark_dir


def load_onnx_model(onnx_path: str) -> onnx.ModelProto:
    """Load ONNX model and convert to version 21.

    :param onnx_path: Path to ONNX model file
    :return: ONNX ModelProto converted to version 21
    """
    import onnx

    from slimonnx.preprocess.version_converter import convert_model_version

    model = onnx.load(onnx_path)
    model = convert_model_version(model, target_opset=21, warn_on_diff=False)
    return model


BENCHMARKS_WITHOUT_BATCH_DIM = (
    "cctsdb_yolo",
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


def check_shape_compatibility(
    inferred_shape: int | list[int], expected_shape: int | list[int]
) -> bool:
    """Check if inferred shape is compatible with expected shape.

    Allows scalar [] to match [1].

    :param inferred_shape: Shape inferred by shape inference
    :param expected_shape: Expected shape from model metadata
    :return: True if shapes are compatible, False otherwise
    """
    return inferred_shape == expected_shape or (inferred_shape == [] and expected_shape == [1])


def infer_shape(
    model, has_batch_dim: bool = True, verbose: bool = False
) -> dict[str, int | list[int]]:
    """Run shape inference on model and validate against expected I/O shapes.

    :param model: ONNX ModelProto
    :param has_batch_dim: Whether model has batch dimension
    :param verbose: Whether to print verbose output during inference
    :return: Dictionary mapping tensor names to inferred shapes
    :raises ValueError: If inferred shapes don't match expected shapes
    """
    from shapeonnx import extract_io_shapes, infer_onnx_shape
    from shapeonnx.utils import (
        convert_constant_to_initializer,
        get_initializers,
        get_input_nodes,
        get_output_nodes,
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

    for output_node in output_nodes:
        output_name = output_node.name
        shape = data_shapes[output_name]
        expected_shape = expected_output_shapes[output_name]
        if not check_shape_compatibility(shape, expected_shape):
            raise ValueError(
                f"Output shape mismatch for '{output_name}': "
                f"inferred shape {shape}, expected shape {expected_shape}"
            )

    return data_shapes  # type: ignore[no-any-return]


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
    except (OSError, KeyError, ValueError):
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
        with Path(instances_csv).open() as file_handle:
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
    except OSError as error:
        raise FileNotFoundError(f"Error reading instances.csv: {error}") from error

    return vnnlib_names


def _get_model_input_info(onnx_path: str) -> tuple[str, list[Any] | None]:
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
    input_dims: list[Any] | None = None
    for inp in model.graph.input:
        if inp.name == input_name:
            input_dims = list(inp.type.tensor_type.shape.dim)
            break

    return input_name, input_dims


def _reshape_array_to_input_dims(arr: np.ndarray, input_dims: list[Any] | None) -> np.ndarray:
    """Reshape flat array to match model input dimensions.

    :param arr: Flat input array
    :param input_dims: ONNX input dimensions
    :return: Reshaped array
    """
    if input_dims is None:
        return arr

    target_shape: list[int] = []
    remaining_size = len(arr)

    for dim in input_dims:
        if dim.dim_value > 0:
            target_shape.append(int(dim.dim_value))
            remaining_size //= dim.dim_value

    if len(target_shape) < len(input_dims):
        target_shape.insert(0, remaining_size)

    return arr.reshape(target_shape)


def _load_from_vnnlib_data(
    vnnlib_data_dir: Path, vnnlib_names: list[str], input_dims: list[Any] | None
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
            except (OSError, KeyError, ValueError, IndexError):
                continue

    return inputs_list


def load_test_inputs(
    onnx_path: str, benchmarks_dir: str = "vnncomp2024_benchmarks", data_dir: str | None = None
) -> list[np.ndarray]:
    """Load test inputs for an ONNX model.

    Tries in order:
    1. Pre-computed data from data/ directory (npz files from calculate_outputs)
    2. VNNLib-derived inputs from vnnlib_data/ directory

    :param onnx_path: Path to ONNX model file
    :param benchmarks_dir: Root benchmarks directory name
    :param data_dir: Root data directory name (default: test_dir/data)
    :return: List of input arrays
    :raises FileNotFoundError: If no test data is available
    """
    if data_dir is None:
        data_dir_path = Path(__file__).parent / "data"
    else:
        data_dir_path = Path(data_dir)

    try:
        benchmark_dir = get_benchmark_dir(onnx_path, benchmarks_dir)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Cannot load test inputs: {error}") from error

    model_name = Path(onnx_path).stem

    # Try pre-computed data first from standalone data/ directory
    from tests.test_benchmarks.benchmark_utils import get_model_benchmark_name

    benchmark_name = get_model_benchmark_name(Path(onnx_path), benchmarks_dir)
    data_file = data_dir_path / benchmark_name / f"{model_name}.npz"
    precomputed_inputs = _load_precomputed_data(data_file)
    if precomputed_inputs:
        return precomputed_inputs

    # Fallback to vnnlib_data
    onnx_rel_path = str(Path(onnx_path).relative_to(benchmark_dir)).replace("\\", "/")
    instances_csv = benchmark_dir / "instances.csv"

    vnnlib_names = _get_vnnlib_names_from_csv(instances_csv, onnx_rel_path)
    if not vnnlib_names:
        raise FileNotFoundError(f"No VNNLib files found for {onnx_rel_path} in instances.csv")

    vnnlib_data_dir = benchmark_dir / "vnnlib_data"
    if not vnnlib_data_dir.exists():
        raise FileNotFoundError(
            f"No vnnlib_data directory found in {benchmark_dir}. Run extract_inputs() first."
        )

    _input_name, input_dims = _get_model_input_info(onnx_path)
    inputs_list = _load_from_vnnlib_data(vnnlib_data_dir, vnnlib_names, input_dims)

    if not inputs_list:
        raise FileNotFoundError(
            f"No valid test inputs found for {onnx_path}. "
            f"Check that vnnlib_data exists and contains valid .npz files."
        )

    return inputs_list
