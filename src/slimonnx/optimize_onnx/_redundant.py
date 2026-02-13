__docformat__ = "restructuredtext"
__all__ = ["_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto


def _skip_redundant_node(
    node: NodeProto, nodes: list[NodeProto], output_nodes: list[ValueInfoProto]
) -> None:
    """Skip a redundant node by rewiring its connections.

    :param node: Node to skip
    :param nodes: All nodes in the graph
    :param output_nodes: Graph output nodes
    """
    for node_j in nodes:
        if node.output[0] in node_j.input:
            for k in range(len(node_j.input)):
                if node_j.input[k] == node.output[0]:
                    node_j.input[k] = node.input[0]

    for output_node_j in output_nodes:
        if node.output[0] == output_node_j.name:
            output_node_j.name = node.input[0]


def _collapse_consecutive_reshapes(nodes: list[NodeProto]) -> list[NodeProto]:
    """Collapse consecutive Reshape nodes.

    :param nodes: List of nodes
    :return: List of nodes with consecutive Reshapes collapsed
    """
    new_nodes = []
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
            for _i, output_name in enumerate(pre_pre_node.output):
                if output_name == pre_node.input[0]:
                    node.input[0] = output_name
            new_nodes.pop()

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    return new_nodes


def _is_redundant_reshape_or_flatten(
    node: NodeProto, data_shapes: dict[str, int | list[int]]
) -> bool:
    """Check if Reshape/Flatten is redundant (no shape change).

    :param node: Node to check
    :param data_shapes: Dictionary of tensor shapes
    :return: True if redundant
    """
    input_shape = data_shapes[node.input[0]]
    output_shape = data_shapes[node.output[0]]
    return input_shape == output_shape


def _is_redundant_arithmetic_op(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> tuple[bool, str | None]:
    """Check if arithmetic operation is redundant (add/sub 0, mul/div 1).

    :param node: Node to check
    :param initializers: Dictionary of initializers
    :return: Tuple of (is_redundant, initializer_name)
    """
    if node.input[1] in initializers:
        initializer_name = node.input[1]
    elif node.input[0] in initializers:
        initializer_name = node.input[0]
    else:
        return False, None

    initializer = initializers[initializer_name]
    array = onnx.numpy_helper.to_array(initializer)

    is_redundant = bool(
        (node.op_type in {"Add", "Sub"} and np.all(array == 0))
        or (node.op_type in {"Mul", "Div"} and np.all(array == 1))
    )

    return is_redundant, initializer_name if is_redundant else None


def _is_redundant_pad(node: NodeProto, initializers: dict[str, TensorProto]) -> bool:
    """Check if Pad operation is redundant (all zeros).

    :param node: Node to check
    :param initializers: Dictionary of initializers
    :return: True if redundant
    """
    initializer = initializers[node.input[1]]
    array = onnx.numpy_helper.to_array(initializer)
    return bool(np.all(array == 0))


def _remove_redundant_operations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
    output_nodes: list[ValueInfoProto],
) -> list[NodeProto]:
    """Remove zero adding, subtracting, multiplying, dividing operations."""
    nodes = _collapse_consecutive_reshapes(nodes)

    new_nodes = []
    for node in nodes:
        if node.op_type in {"Reshape", "Flatten"}:
            if _is_redundant_reshape_or_flatten(node, data_shapes):
                if node.op_type == "Reshape":
                    del initializers[node.input[1]]
                _skip_redundant_node(node, nodes, output_nodes)
                continue

        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            is_redundant, initializer_name = _is_redundant_arithmetic_op(node, initializers)
            if is_redundant and initializer_name is not None:
                del initializers[initializer_name]
                _skip_redundant_node(node, nodes, output_nodes)
                continue

        elif node.op_type == "Pad" and _is_redundant_pad(node, initializers):
            del initializers[node.input[1]]
            _skip_redundant_node(node, nodes, output_nodes)
            continue

        new_nodes.append(node)

    return new_nodes
