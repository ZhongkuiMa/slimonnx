"""Remove redundant identity-like operations from ONNX graphs."""

__docformat__ = "restructuredtext"
__all__ = ["_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto

from slimonnx.optimize_onnx._reshape import _collapse_consecutive_reshapes


def _rewire_redundant_node(
    node: NodeProto,
    nodes: list[NodeProto],
    output_nodes: list[ValueInfoProto],
    replacement: str | None = None,
) -> None:
    """Rewire downstream consumers around a redundant identity-like node.

    Replaces ``node.output[0]`` everywhere it is consumed (in other nodes'
    inputs and in the graph output list) with ``replacement`` (or
    ``node.input[0]`` by default). The node
    itself remains in the list; callers are responsible for dropping it
    afterwards. Was previously named ``_skip_redundant_node`` -- the
    verb here is genuinely rewiring, not skipping.

    :param node: Identity-like node to bypass.

    :param nodes: All nodes in the graph.

    :param output_nodes: Graph output value-infos that may name
        ``node.output[0]`` and need redirecting.

    :param replacement: Explicit source value for commutative arithmetic
        whose identity initializer occupies ``node.input[0]``.
    """
    redundant_output = node.output[0]
    replacement = node.input[0] if replacement is None else replacement
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
    right_name = node.input[1]
    if right_name in initializers:
        right = onnx.numpy_helper.to_array(initializers[right_name])
        if (node.op_type in {"Add", "Sub"} and np.all(right == 0)) or (
            node.op_type in {"Mul", "Div"} and np.all(right == 1)
        ):
            return True, right_name

    left_name = node.input[0]
    if left_name in initializers and node.op_type in {"Add", "Mul"}:
        left = onnx.numpy_helper.to_array(initializers[left_name])
        if (node.op_type == "Add" and np.all(left == 0)) or (
            node.op_type == "Mul" and np.all(left == 1)
        ):
            return True, left_name

    return False, None


def _is_redundant_pad(node: NodeProto, initializers: dict[str, TensorProto]) -> bool:
    """Check if Pad operation is redundant (all zeros).

    :param node: Node to check.

    :param initializers: Dictionary of initializers.

    :return: True if redundant
    """
    initializer = initializers[node.input[1]]
    array = onnx.numpy_helper.to_array(initializer)
    return bool(np.all(array == 0))


def _is_redundant_cast(node: NodeProto, data_types: dict[str, int]) -> bool:
    """Return whether ``Cast`` preserves an already-known element type."""
    source_type = data_types.get(node.input[0])
    target_type = next((int(attr.i) for attr in node.attribute if attr.name == "to"), None)
    return source_type is not None and target_type == source_type


def _remove_redundant_operations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
    output_nodes: list[ValueInfoProto],
    data_types: dict[str, int] | None = None,
) -> list[NodeProto]:
    """Remove identity-like operations proved redundant by shape/value/type."""
    graph_output_names = {output.name for output in output_nodes}
    nodes = _collapse_consecutive_reshapes(nodes, initializers, graph_output_names)
    data_types = {} if data_types is None else data_types
    removable_initializers: set[str] = set()

    new_nodes = []
    for node in nodes:
        if node.domain not in {"", "ai.onnx"} or any(
            output in graph_output_names for output in node.output
        ):
            new_nodes.append(node)
            continue

        if node.op_type in {"Reshape", "Flatten"}:
            if _is_redundant_reshape_or_flatten(node, data_shapes):
                if node.op_type == "Reshape" and node.input[1] in initializers:
                    removable_initializers.add(node.input[1])
                _rewire_redundant_node(node, nodes, output_nodes)
                continue

        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            is_redundant, initializer_name = _is_redundant_arithmetic_op(node, initializers)
            if is_redundant and initializer_name is not None:
                replacement = node.input[1] if node.input[0] == initializer_name else node.input[0]
                removable_initializers.add(initializer_name)
                _rewire_redundant_node(node, nodes, output_nodes, replacement)
                continue

        elif node.op_type == "Pad" and _is_redundant_pad(node, initializers):
            removable_initializers.add(node.input[1])
            _rewire_redundant_node(node, nodes, output_nodes)
            continue

        elif node.op_type == "Cast" and _is_redundant_cast(node, data_types):
            _rewire_redundant_node(node, nodes, output_nodes)
            continue

        new_nodes.append(node)

    live_values = {
        input_name for node in new_nodes for input_name in node.input if input_name
    } | graph_output_names
    for initializer_name in removable_initializers - live_values:
        initializers.pop(initializer_name, None)

    return new_nodes
