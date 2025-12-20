"""Conv + BatchNorm fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_conv_bn",
    "detect_bn_conv",
    "detect_convtranspose_bn",
    "detect_bn_convtranspose",
]

from onnx import NodeProto, TensorProto


def _is_consecutive_nodes(
    first_node: NodeProto, second_node: NodeProto, nodes: list[NodeProto]
) -> bool:
    """Check if second_node immediately follows first_node with no other consumers.

    :param first_node: First node in pattern
    :param second_node: Second node in pattern
    :param nodes: All nodes in graph
    :return: True if nodes are consecutive with no branching
    """
    # Check if first_node's output feeds into second_node's input
    if first_node.output[0] != second_node.input[0]:
        return False

    # Check if first_node's output has only one consumer (second_node)
    first_output = first_node.output[0]
    consumer_count = sum(1 for node in nodes if first_output in node.input)

    return consumer_count == 1


def detect_conv_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect Conv + BatchNormalization fusion pattern.

    Pattern: Conv -> BatchNormalization
    Can be fused into single Conv with modified weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances with Conv and BN node info
    """
    instances = []

    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i + 1]

        # Check pattern: Conv followed by BatchNormalization
        if curr_node.op_type != "Conv" or next_node.op_type != "BatchNormalization":
            continue

        # Check if they are consecutive with no branching
        if not _is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if Conv has constant weight (must be in initializers)
        if len(curr_node.input) < 2 or curr_node.input[1] not in initializers:
            continue

        # Check if BN has all required parameters in initializers
        bn_inputs = next_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [
            bn_inputs[1],
            bn_inputs[2],
            bn_inputs[3],
            bn_inputs[4],
        ]  # scale, B, mean, var
        if not all(param in initializers for param in required_params):
            continue

        instances.append(
            {
                "conv_node": curr_node.name,
                "bn_node": next_node.name,
                "conv_weight": curr_node.input[1],
                "bn_scale": bn_inputs[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + Conv fusion pattern.

    Pattern: BatchNormalization -> Conv
    Can be fused into single Conv with modified weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances with BN and Conv node info
    """
    instances = []

    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i + 1]

        # Check pattern: BatchNormalization followed by Conv
        if curr_node.op_type != "BatchNormalization" or next_node.op_type != "Conv":
            continue

        # Check if they are consecutive with no branching
        if not _is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if BN has all required parameters in initializers
        bn_inputs = curr_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        # Check if Conv has constant weight
        if len(next_node.input) < 2 or next_node.input[1] not in initializers:
            continue

        instances.append(
            {
                "bn_node": curr_node.name,
                "conv_node": next_node.name,
                "bn_scale": bn_inputs[1],
                "conv_weight": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_convtranspose_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect ConvTranspose + BatchNormalization fusion pattern.

    Pattern: ConvTranspose -> BatchNormalization
    Can be fused into single ConvTranspose with modified weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances with ConvTranspose and BN node info
    """
    instances = []

    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i + 1]

        # Check pattern: ConvTranspose followed by BatchNormalization
        if curr_node.op_type != "ConvTranspose" or next_node.op_type != "BatchNormalization":
            continue

        # Check if they are consecutive with no branching
        if not _is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if ConvTranspose has constant weight
        if len(curr_node.input) < 2 or curr_node.input[1] not in initializers:
            continue

        # Check if BN has all required parameters
        bn_inputs = next_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        instances.append(
            {
                "convtranspose_node": curr_node.name,
                "bn_node": next_node.name,
                "convtranspose_weight": curr_node.input[1],
                "bn_scale": bn_inputs[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_convtranspose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + ConvTranspose fusion pattern.

    Pattern: BatchNormalization -> ConvTranspose
    Can be fused into single ConvTranspose with modified weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances with BN and ConvTranspose node info
    """
    instances = []

    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i + 1]

        # Check pattern: BatchNormalization followed by ConvTranspose
        if curr_node.op_type != "BatchNormalization" or next_node.op_type != "ConvTranspose":
            continue

        # Check if they are consecutive with no branching
        if not _is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if BN has all required parameters
        bn_inputs = curr_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        # Check if ConvTranspose has constant weight
        if len(next_node.input) < 2 or next_node.input[1] not in initializers:
            continue

        instances.append(
            {
                "bn_node": curr_node.name,
                "convtranspose_node": next_node.name,
                "bn_scale": bn_inputs[1],
                "convtranspose_weight": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances
