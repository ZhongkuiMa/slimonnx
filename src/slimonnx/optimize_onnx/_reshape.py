"""Reshape optimization operations."""

__docformat__ = "restructuredtext"
__all__ = ["_collapse_consecutive_reshapes", "_resolve_reshape_negative_one"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def _reshape_target_is_source_independent(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> bool:
    """Return whether the target shape has no input-relative dimensions."""
    shape = initializers.get(node.input[1])
    if shape is None:
        return False
    allowzero = next((int(attr.i) for attr in node.attribute if attr.name == "allowzero"), 0)
    target = onnx.numpy_helper.to_array(shape)
    return bool(allowzero == 1 or not np.any(target == 0))


def _collapse_consecutive_reshapes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    graph_output_names: set[str],
) -> list[NodeProto]:
    """Collapse adjacent ``Reshape -> Reshape`` pairs to a single Reshape.

    The first reshape is redundant only when the second consumes it
    exclusively, the intermediate value is not public, and the second target
    shape is static and independent of its immediate input dimensions. A
    default-``allowzero`` target containing zero is not independent: zero
    copies a dimension from the first reshape's output, so bypassing that
    reshape can change values or make the graph invalid.

    :param nodes: Model nodes.

    :param initializers: Initializers used to prove the second target shape.

    :param graph_output_names: Observable graph values that cannot be bypassed.

    :return: New node list with redundant intermediate Reshapes removed.
    :raises ValueError: If a Reshape node violates the expected 2-input /
        1-output shape contract.
    """
    consumer_counts: dict[str, int] = {}
    for candidate in nodes:
        for input_name in candidate.input:
            consumer_counts[input_name] = consumer_counts.get(input_name, 0) + 1

    new_nodes: list[NodeProto] = []
    for node in nodes:
        previous = new_nodes[-1] if new_nodes else None
        if (
            previous is not None
            and node.op_type == "Reshape"
            and previous.op_type == "Reshape"
            and previous.domain in {"", "ai.onnx"}
            and node.domain in {"", "ai.onnx"}
        ):
            if (
                len(previous.input) != 2
                or len(previous.output) != 1
                or len(node.input) != 2
                or len(node.output) != 1
            ):
                raise ValueError(
                    f"Invalid Reshape node structure: {previous.name} "
                    f"inputs={len(previous.input)}, outputs={len(previous.output)}, "
                    f"{node.name} inputs={len(node.input)}, outputs={len(node.output)}. "
                    "Expected 2 inputs and 1 output for both nodes."
                )
            intermediate = previous.output[0]
            can_collapse = (
                node.input[0] == intermediate
                and consumer_counts.get(intermediate) == 1
                and intermediate not in graph_output_names
                and _reshape_target_is_source_independent(node, initializers)
            )
            if can_collapse:
                node.input[0] = previous.input[0]
                new_nodes.pop()
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
