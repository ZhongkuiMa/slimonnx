"""Common utilities for pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "has_constant_weight",
    "is_consecutive_nodes",
    "validate_bn_inputs",
]

from onnx import NodeProto, TensorProto


def is_consecutive_nodes(
    first_node: NodeProto,
    second_node: NodeProto,
    nodes: list[NodeProto],
) -> bool:
    """Check if second_node immediately follows first_node with no branching.

    Verifies that:
    1. first_node output connects to second_node input
    2. No other nodes consume first_node's output (no branching)

    :param first_node: First node in potential pattern
    :param second_node: Second node in potential pattern
    :param nodes: All nodes in graph
    :return: True if nodes are consecutive without branching
    """
    # Defensive check for empty output/input (from depthwise_conv.py)
    if not first_node.output or not second_node.input:
        return False

    # Check if outputs/inputs match
    if first_node.output[0] != second_node.input[0]:
        return False

    # Check for branching - no other node should use first_node's output
    first_output = first_node.output[0]
    for node in nodes:
        if node == second_node:
            continue
        if first_output in node.input:
            return False  # Branching detected

    return True


def validate_bn_inputs(
    bn_node: NodeProto,
    initializers: dict[str, TensorProto],
) -> bool:
    """Validate BatchNormalization node has all required parameters.

    BN requires 5 inputs: [X, scale, B, mean, var]
    This checks that the node has 5 inputs and all constant parameters
    (scale, B, mean, var) are available in initializers.

    :param bn_node: BatchNormalization node to validate
    :param initializers: Model initializers dictionary
    :return: True if BN has all required parameters
    """
    # Check input count
    if len(bn_node.input) < 5:
        return False

    # Check that all BN parameters are in initializers
    # bn_node.input[0] is the variable input (X)
    # bn_node.input[1:5] are constant parameters
    required_params = bn_node.input[1:5]  # scale, bias, mean, var
    return all(param in initializers for param in required_params)


def has_constant_weight(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    weight_index: int = 1,
) -> bool:
    """Check if node has constant weight at specified input index.

    :param node: Node to check
    :param initializers: Model initializers dictionary
    :param weight_index: Index of weight input (default: 1)
    :return: True if node has weight in initializers
    """
    if len(node.input) <= weight_index:
        return False
    return node.input[weight_index] in initializers
