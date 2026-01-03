"""Reshape optimization operations."""

__docformat__ = "restructuredtext"
__all__ = ["_resolve_reshape_negative_one"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def _resolve_reshape_negative_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
) -> list[NodeProto]:
    """Replace -1 in Reshape shape tensors with concrete values.

    When shape inference has determined the exact output shape of a Reshape
    operation, update the shape tensor initializer to use concrete values
    instead of -1.

    :param nodes: Model nodes
    :param initializers: Model initializers (modified in-place)
    :param data_shapes: Inferred shapes from shape inference
    :return: Original nodes (unchanged, only initializers are modified)
    """
    for node in nodes:
        if node.op_type != "Reshape":
            continue

        if len(node.input) < 2:
            continue

        shape_input_name = node.input[1]

        # Check if shape is an initializer
        if shape_input_name not in initializers:
            continue

        # Get the shape tensor values
        shape_tensor = initializers[shape_input_name]
        shape_values = onnx.numpy_helper.to_array(shape_tensor)

        # Check if -1 is present in shape
        if -1 not in shape_values:
            continue

        # Check if output shape is known (no zeros indicating unknown)
        output_name = node.output[0]
        if output_name not in data_shapes:
            continue

        output_shape = data_shapes[output_name]
        # Handle both int and list[int] cases
        if isinstance(output_shape, int):
            output_shape = [output_shape]

        if 0 in output_shape:
            # Output shape is dynamic/unknown, cannot resolve
            continue

        # Create new shape tensor with concrete values
        new_shape_values = np.array(output_shape, dtype=np.int64)
        new_initializer = onnx.numpy_helper.from_array(new_shape_values, shape_input_name)
        initializers[shape_input_name] = new_initializer

    return nodes
