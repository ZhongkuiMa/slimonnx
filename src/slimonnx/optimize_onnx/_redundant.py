"""Remove redundant identity-like operations from ONNX graphs."""

__docformat__ = "restructuredtext"
__all__ = ["_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto

from slimonnx.optimize_onnx._reshape import _collapse_consecutive_reshapes


def _rewire_redundant_node(
    node: NodeProto, nodes: list[NodeProto], output_nodes: list[ValueInfoProto]
) -> None:
    """Rewire downstream consumers around a redundant identity-like node.

    Replaces ``node.output[0]`` everywhere it is consumed (in other nodes'
    inputs and in the graph output list) with ``node.input[0]``. The node
    itself remains in the list; callers are responsible for dropping it
    afterwards. Was previously named ``_skip_redundant_node`` -- the
    verb here is genuinely rewiring, not skipping.

    :param node: Identity-like node to bypass.

    :param nodes: All nodes in the graph.

    :param output_nodes: Graph output value-infos that may name
        ``node.output[0]`` and need redirecting.
    """
    redundant_output = node.output[0]
    replacement = node.input[0]
    for node_j in nodes:
        if redundant_output not in node_j.input:
            continue
        for k, input_name in enumerate(node_j.input):
            if input_name == redundant_output:
                node_j.input[k] = replacement

    for output_node_j in output_nodes:
        if output_node_j.name == redundant_output:
            output_node_j.name = replacement


def _is_redundant_reshape_or_flatten(
    node: NodeProto, data_shapes: dict[str, int | list[int]]
) -> bool:
    """Check if Reshape/Flatten is redundant (no shape change).

    :param node: Node to check.

    :param data_shapes: Dictionary of tensor shapes.

    :return: True if redundant
    """
    input_shape = data_shapes[node.input[0]]
    output_shape = data_shapes[node.output[0]]
    return input_shape == output_shape


def _is_redundant_arithmetic_op(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> tuple[bool, str | None]:
    """Check if arithmetic operation is redundant (add/sub 0, mul/div 1).

    :param node: Node to check.

    :param initializers: Dictionary of initializers.

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

    :param node: Node to check.

    :param initializers: Dictionary of initializers.

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
                if node.op_type == "Reshape" and node.input[1] in initializers:
                    del initializers[node.input[1]]
                _rewire_redundant_node(node, nodes, output_nodes)
                continue

        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            is_redundant, initializer_name = _is_redundant_arithmetic_op(node, initializers)
            if is_redundant and initializer_name is not None:
                del initializers[initializer_name]
                _rewire_redundant_node(node, nodes, output_nodes)
                continue

        elif node.op_type == "Pad" and _is_redundant_pad(node, initializers):
            del initializers[node.input[1]]
            _rewire_redundant_node(node, nodes, output_nodes)
            continue

        new_nodes.append(node)

    return new_nodes
