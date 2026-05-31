"""Reshape optimization operations."""

__docformat__ = "restructuredtext"
__all__ = ["_collapse_consecutive_reshapes", "_resolve_reshape_negative_one"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def _collapse_consecutive_reshapes(nodes: list[NodeProto]) -> list[NodeProto]:
    """Collapse adjacent ``Reshape -> Reshape`` pairs to a single Reshape.

    When two Reshape nodes appear back to back with no other consumer in
    between, the first reshape is redundant: the second reshape already
    targets the final shape. This pass rewires the second Reshape to read
    directly from the predecessor of the first.

    :param nodes: Model nodes.

    :return: New node list with redundant intermediate Reshapes removed.
    :raises ValueError: If a Reshape node violates the expected 2-input /
        1-output shape contract.
    """
    new_nodes: list[NodeProto] = []
    pre_pre_node = None
    pre_node = None

    for node in nodes:
        if (
            pre_node is not None
            and pre_pre_node is not None
            and node.op_type == "Reshape"
            and pre_node.op_type == "Reshape"
        ):
            if len(pre_node.input) != 2 or len(pre_node.output) != 1 or len(node.input) != 2:
                raise ValueError(
                    f"Invalid Reshape node structure: {pre_node.name} "
                    f"inputs={len(pre_node.input)}, outputs={len(pre_node.output)}, "
                    f"{node.name} inputs={len(node.input)}. "
                    "Expected 2 inputs and 1 output."
                )
            for output_name in pre_pre_node.output:
                if output_name == pre_node.input[0]:
                    node.input[0] = output_name
            new_nodes.pop()

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    return new_nodes


def _resolve_reshape_negative_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
) -> list[NodeProto]:
    """Replace -1 in Reshape shape tensors with concrete values.

    When shape inference has determined the exact output shape of a Reshape
    operation, update the shape tensor initializer to use concrete values
    instead of -1.

    :param nodes: Model nodes.

    :param initializers: Model initializers (modified in-place).

    :param data_shapes: Inferred shapes from shape inference.

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
