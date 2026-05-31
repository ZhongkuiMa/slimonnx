"""Utility functions for ONNX model manipulation and analysis."""

__docformat__ = "restructuredtext"
__all__ = [
    "EXTRACT_ATTRS_MAP",
    "clear_onnx_docstring",
    "compare_outputs",
    "convert_constant_to_initializer",
    "extract_nodes",
    "generate_random_inputs",
    "get_initializers",
    "get_input_nodes",
    "get_next_nodes_mapping",
    "get_output_nodes",
    "has_single_consumer",
    "load_test_data_from_file",
    "reformat_io_shape",
]

from pathlib import Path

import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto
from shapeonnx.onnx_attrs import EXTRACT_ATTRS_MAP
from shapeonnx.utils import (
    _reformat_io_shape as reformat_io_shape,
)
from shapeonnx.utils import (
    convert_constant_to_initializer,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)

# ONNX TensorProto element-type -> numpy dtype. Built once at import time so
# generate_random_inputs() does not rebuild it per call. Keyed on the
# TensorProto enum values to keep the source of truth at the ONNX spec.
_TENSOR_PROTO_TO_NUMPY_DTYPE: dict[int, type] = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
}

# Floating-point numpy dtypes that get sampled from a standard normal
# distribution; everything else gets uniform integer sampling.
_FLOAT_NUMPY_DTYPES: frozenset[type] = frozenset({np.float16, np.float32, np.float64})


def clear_onnx_docstring(model: ModelProto) -> ModelProto:
    """Clear all doc strings from ONNX model nodes.

    :param model: ONNX model.

    :return: Model with cleared docstrings
    """
    for node in model.graph.node:
        node.doc_string = ""
    return model


def has_single_consumer(
    producer: NodeProto,
    consumer: NodeProto,
    nodes: list[NodeProto],
) -> bool:
    """Check if ``consumer`` is the only node reading ``producer``'s first output.

    Linear fusion passes (Conv-BN, Gemm-BN, etc.) may rewrite the producer
    only when no other node depends on its first output -- otherwise the
    fusion silently drops a forward edge from the computation graph.

    :param producer: Candidate upstream node whose ``output[0]`` is checked.

    :param consumer: Candidate downstream node that should be the exclusive
        reader of ``producer.output[0]``.

    :param nodes: Full graph node list, scanned for other consumers.

    :return: ``True`` if no node other than ``consumer`` reads
        ``producer.output[0]``.
    """
    prod_output = producer.output[0]
    return all(not (prod_output in node.input and node is not consumer) for node in nodes)


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

    :param model: ONNX model.

    :param has_batch_dim: Whether the model has batch dimension.

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

    .. note::
        Side effect: nodes whose ``name`` is the empty string (which can
        happen after ONNX version conversion) are assigned a synthesized
        name of the form ``"<op_type>_<n>"`` so they can be used as map
        keys. This mutation is required for the returned mapping to be
        coherent; callers that need to preserve the original node names
        should snapshot them before calling.

    :param nodes: List of ONNX nodes.

    :return: Dictionary mapping node names to list of next node names
    """
    empty_name_counter = 0
    output_name_to_node_name: dict[str, str] = {}
    for node in nodes:
        if node.name == "":
            # ONNX version conversion can leave nodes with empty names; assign
            # a deterministic synthetic name so map keys exist.
            node.name = f"{node.op_type}_{empty_name_counter}"
            empty_name_counter += 1
        for output_name in node.output:
            output_name_to_node_name[output_name] = node.name

    next_nodes_mapping: dict[str, list[str]] = {node.name: [] for node in nodes}
    for node in nodes:
        for input_name in node.input:
            producer_name = output_name_to_node_name.get(input_name)
            if producer_name is not None:
                next_nodes_mapping[producer_name].append(node.name)

    return next_nodes_mapping


def generate_random_inputs(
    model: ModelProto,
    num_samples: int = 1,
) -> list[dict[str, np.ndarray]]:
    """Generate random inputs matching model signature.

    :param model: ONNX model.

    :param num_samples: Number of input samples to generate.

    :return: List of input dictionaries
    """
    inputs_list = []
    rng = np.random.default_rng()
    initializer_names = {init.name for init in model.graph.initializer}

    for _ in range(num_samples):
        input_dict = {}
        for input_info in model.graph.input:
            # Initializer-backed inputs are not real model inputs.
            if input_info.name in initializer_names:
                continue

            shape = tuple(
                d.dim_value if d.HasField("dim_value") else 1
                for d in input_info.type.tensor_type.shape.dim
            )
            elem_type = input_info.type.tensor_type.elem_type
            dtype = _TENSOR_PROTO_TO_NUMPY_DTYPE.get(elem_type, np.float32)

            if dtype in _FLOAT_NUMPY_DTYPES:
                input_array = rng.standard_normal(shape).astype(dtype)
            else:
                input_array = rng.integers(0, 10, size=shape).astype(dtype)

            input_dict[input_info.name] = input_array

        inputs_list.append(input_dict)

    return inputs_list


def load_test_data_from_file(data_path: str) -> list[dict[str, np.ndarray]]:
    """Load test input-output data from .npy or .npz file.

    :param data_path: Path to test data file.

    :return: List of input dictionaries
    :raises FileNotFoundError: If data file not found.

    :raises ValueError: If unsupported file format.

    """
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

    :param outputs1: First output dictionary.

    :param outputs2: Second output dictionary.

    :param rtol: Relative tolerance for comparison.

    :param atol: Absolute tolerance for comparison.

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
