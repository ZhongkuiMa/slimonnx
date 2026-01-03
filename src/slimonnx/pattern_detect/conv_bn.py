"""Conv + BatchNorm fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_bn_conv",
    "detect_bn_conv_transpose",
    "detect_conv_bn",
    "detect_conv_transpose_bn",
]

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect.utils import (
    has_constant_weight,
    is_consecutive_nodes,
    validate_bn_inputs,
)


def detect_conv_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if Conv has constant weight
        if not has_constant_weight(curr_node, initializers):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(next_node, initializers):
            continue

        instances.append(
            {
                "conv_node": curr_node.name,
                "bn_node": next_node.name,
                "conv_weight": curr_node.input[1],
                "bn_scale": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(curr_node, initializers):
            continue

        # Check if Conv has constant weight
        if not has_constant_weight(next_node, initializers):
            continue

        instances.append(
            {
                "bn_node": curr_node.name,
                "conv_node": next_node.name,
                "bn_scale": curr_node.input[1],
                "conv_weight": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_conv_transpose_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if ConvTranspose has constant weight
        if not has_constant_weight(curr_node, initializers):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(next_node, initializers):
            continue

        instances.append(
            {
                "conv_transpose_node": curr_node.name,
                "bn_node": next_node.name,
                "conv_transpose_weight": curr_node.input[1],
                "bn_scale": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_conv_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(curr_node, initializers):
            continue

        # Check if ConvTranspose has constant weight
        if not has_constant_weight(next_node, initializers):
            continue

        instances.append(
            {
                "bn_node": curr_node.name,
                "conv_transpose_node": next_node.name,
                "bn_scale": curr_node.input[1],
                "conv_transpose_weight": next_node.input[1],
                "can_fuse": True,
            }
        )

    return instances
