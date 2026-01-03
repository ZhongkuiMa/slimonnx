"""Utility functions for ONNX model manipulation and analysis."""

__docformat__ = "restructuredtext"
__all__ = [
    "EXTRACT_ATTR_MAP",
    "clear_onnx_docstring",
    "compare_outputs",
    "convert_constant_to_initializer",
    "extract_nodes",
    "generate_random_inputs",
    "get_initializers",
    "get_input_nodes",
    "get_next_nodes_mapping",
    "get_output_nodes",
    "load_test_data_from_file",
    "reformat_io_shape",
]

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto

EXTRACT_ATTR_MAP = {
    0: lambda x: None,  # UNDEFINED
    1: lambda x: x.f,  # FLOAT
    2: lambda x: x.i,  # INT
    3: lambda x: x.s.decode("utf-8"),  # STRING
    4: lambda x: onnx.numpy_helper.to_array(x.t),  # TENSOR
    5: lambda x: x.g,  # GRAPH
    6: lambda x: tuple(x.floats),  # FLOATS
    7: lambda x: tuple(x.ints),  # INTS
    8: lambda x: None,  # STRINGS
    9: lambda x: None,  # TENSORS
    10: lambda x: None,  # GRAPHS
    11: lambda x: None,  # SPARSE_TENSOR
}


def clear_onnx_docstring(model: ModelProto) -> ModelProto:
    """Clear all doc strings from ONNX model nodes.

    :param model: ONNX model
    :return: Model with cleared docstrings
    """
    for node in model.graph.node:
        node.doc_string = ""
    return model


def reformat_io_shape(node: ValueInfoProto, has_batch_dim: bool = True) -> list[int]:
    """Extract shape from ValueInfoProto node."""
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]
    if has_batch_dim:
        # Allow scalar outputs [] - they don't need batch dimension validation
        # (e.g., outputs reduced via Squeeze operations)
        if len(shape) == 0:
            return shape
        if len(shape) < 2:
            raise ValueError(
                f"There should have been a batch dimension. "
                f"Node {node.name} has invalid shape {shape}."
            )
        if shape[0] != 1:
            shape[0] = 1

    return shape


def get_input_nodes(
    model: ModelProto,
    initializers: dict[str, TensorProto],
    has_batch_dim: bool = True,
) -> list[ValueInfoProto]:
    """Get input nodes from ONNX model.

    :param model: ONNX model
    :param initializers: Dictionary of initializers
    :param has_batch_dim: Whether the model has batch dimension
    :return: List of input nodes
    """
    # Exclude initializers from inputs because sometimes the initializers are also
    # included in the inputs
    nodes = []
    for input_i in model.graph.input:
        if input_i.name not in initializers:
            shape = reformat_io_shape(input_i, has_batch_dim)
            node = onnx.helper.make_tensor_value_info(
                name=input_i.name,
                elem_type=input_i.type.tensor_type.elem_type,
                shape=shape,
            )
            nodes.append(node)

    return nodes


def get_output_nodes(model: ModelProto, has_batch_dim: bool = True) -> list[ValueInfoProto]:
    """Get output nodes from ONNX model.

    :param model: ONNX model
    :param has_batch_dim: Whether the model has batch dimension
    :return: List of output nodes
    """
    nodes = []
    for output_i in model.graph.output:
        shape = reformat_io_shape(output_i, has_batch_dim)
        node = onnx.helper.make_tensor_value_info(
            name=output_i.name,
            elem_type=output_i.type.tensor_type.elem_type,
            shape=shape,
        )
        nodes.append(node)

    return nodes


def get_initializers(model: ModelProto) -> dict[str, TensorProto]:
    """Get initializers from ONNX model.

    :param model: ONNX model
    :return: Dictionary of initializers
    """
    return {initializer.name: initializer for initializer in model.graph.initializer}


def convert_constant_to_initializer(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    """Convert Constant nodes to initializers.

    This is critical for shape inference and optimization.
    Constant nodes are converted to initializers and removed from node list.

    :param nodes: List of ONNX nodes
    :param initializers: Initializers dictionary to update (modified in-place)
    :return: List of nodes with Constant nodes removed
    """
    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            # Extract tensor from Constant node
            np_array = onnx.numpy_helper.to_array(node.attribute[0].t)
            initializer = onnx.numpy_helper.from_array(np_array, node.output[0])
            initializers[node.output[0]] = initializer
            continue
        new_nodes.append(node)
    return new_nodes


def extract_nodes(
    model: ModelProto,
    has_batch_dim: bool = True,
) -> tuple[list[ValueInfoProto], list[ValueInfoProto], list[NodeProto], dict[str, TensorProto]]:
    """Extract and preprocess nodes from ONNX model.

    This is a helper function that:
    1. Extracts initializers
    2. Converts Constant nodes to initializers (critical for shape inference)
    3. Extracts input nodes (excluding initializers)
    4. Extracts output nodes

    :param model: ONNX model
    :param has_batch_dim: Whether the model has batch dimension
    :return: Tuple of (input_nodes, output_nodes, nodes, initializers)
    """
    # Get initializers
    initializers = get_initializers(model)

    # Convert Constant nodes to initializers
    nodes = list(model.graph.node)
    nodes = convert_constant_to_initializer(nodes, initializers)

    # Get input and output nodes
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)

    return input_nodes, output_nodes, nodes, initializers


def get_next_nodes_mapping(nodes: list[NodeProto]) -> dict[str, list[str]]:
    """Get the mapping from each node to its next nodes.

    :param nodes: List of ONNX nodes
    :return: Dictionary mapping node names to list of next node names
    """
    empty_name_counter = 0
    name_and_output_name_mapping = {}
    for node in nodes:
        for output_name in node.output:
            if node.name == "":
                # Sometimes, there will be a node with empty string name.
                # This is caused during the ONNX version conversion.
                node.name = f"{node.op_type}_{empty_name_counter}"
                empty_name_counter += 1
            name_and_output_name_mapping[output_name] = node.name

    next_nodes_mapping: dict[str, list[str]] = {node.name: [] for node in nodes}
    for node in nodes:
        for input_name in node.input:
            if input_name in name_and_output_name_mapping:
                next_nodes_mapping[name_and_output_name_mapping[input_name]].append(node.name)

    return next_nodes_mapping


def generate_random_inputs(
    model: ModelProto,
    num_samples: int = 1,
) -> list[dict[str, np.ndarray]]:
    """Generate random inputs matching model signature.

    :param model: ONNX model
    :param num_samples: Number of input samples to generate
    :return: List of input dictionaries
    """
    inputs_list = []
    rng = np.random.default_rng()

    for _ in range(num_samples):
        input_dict = {}
        for input_info in model.graph.input:
            # Skip if this is an initializer (not a true input)
            initializer_names = {init.name for init in model.graph.initializer}
            if input_info.name in initializer_names:
                continue

            # Get shape
            shape = tuple(
                d.dim_value if d.HasField("dim_value") else 1
                for d in input_info.type.tensor_type.shape.dim
            )

            # Get dtype
            dtype_map = {
                1: np.float32,  # FLOAT
                2: np.uint8,  # UINT8
                3: np.int8,  # INT8
                6: np.int32,  # INT32
                7: np.int64,  # INT64
                10: np.float16,  # FLOAT16
                11: np.float64,  # DOUBLE
            }
            elem_type = input_info.type.tensor_type.elem_type
            dtype = dtype_map.get(elem_type, np.float32)

            # Generate random input
            if dtype in (np.float32, np.float64, np.float16):
                input_array = rng.standard_normal(shape).astype(dtype)
            else:
                input_array = rng.integers(0, 10, size=shape).astype(dtype)

            input_dict[input_info.name] = input_array

        inputs_list.append(input_dict)

    return inputs_list


def load_test_data_from_file(data_path: str) -> list[dict[str, np.ndarray]]:
    """Load test input-output data from .npy or .npz file.

    :param data_path: Path to test data file
    :return: List of input dictionaries
    :raises FileNotFoundError: If data file not found
    :raises ValueError: If unsupported file format
    """
    from pathlib import Path

    if not Path(data_path).exists():
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    if data_path.endswith(".npy"):
        data = np.load(data_path)
        if len(data.shape) == 1:
            return [{"input": data}]
        return [{"input": data[i]} for i in range(data.shape[0])]

    if data_path.endswith(".npz"):
        data = np.load(data_path)
        if "inputs" in data:
            inputs_data = data["inputs"]
            return [{"input": inputs_data[i]} for i in range(inputs_data.shape[0])]
        if "X" in data:
            x_data = data["X"]
            return [{"input": x_data[i]} for i in range(x_data.shape[0])]
        key = next(iter(data.keys()))
        arr = data[key]
        return [{"input": arr[i]} for i in range(arr.shape[0])]

    raise ValueError(f"Unsupported file format: {data_path}")


def compare_outputs(
    outputs1: dict[str, np.ndarray],
    outputs2: dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[bool, list[dict]]:
    """Compare two output dictionaries.

    :param outputs1: First output dictionary
    :param outputs2: Second output dictionary
    :param rtol: Relative tolerance for comparison
    :param atol: Absolute tolerance for comparison
    :return: Tuple of (all_match, mismatch_details)
    """
    mismatches = []

    # Check if all keys match
    keys1 = set(outputs1.keys())
    keys2 = set(outputs2.keys())

    if keys1 != keys2:
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            mismatches.append(
                {
                    "type": "missing_key",
                    "message": f"Keys missing in outputs2: {missing_in_2}",
                }
            )
        if missing_in_1:
            mismatches.append(
                {
                    "type": "missing_key",
                    "message": f"Keys missing in outputs1: {missing_in_1}",
                }
            )
        return False, mismatches

    # Compare each output
    for key in outputs1:
        out1 = outputs1[key]
        out2 = outputs2[key]

        # Check shape match
        if out1.shape != out2.shape:
            mismatches.append(
                {
                    "key": key,
                    "type": "shape_mismatch",
                    "shape1": str(out1.shape),
                    "shape2": str(out2.shape),
                    "message": f"Shape mismatch for '{key}': {out1.shape} vs {out2.shape}",
                }
            )
            continue

        # Check numerical match
        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(out1 - out2))
            mismatches.append(
                {
                    "key": key,
                    "type": "value_mismatch",
                    "max_diff": str(float(max_diff)),
                    "rtol": str(rtol),
                    "atol": str(atol),
                    "message": f"Values differ for '{key}': max_diff={max_diff:.2e}",
                }
            )

    return len(mismatches) == 0, mismatches
