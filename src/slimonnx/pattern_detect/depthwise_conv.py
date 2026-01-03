"""Depthwise convolution pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_bn_depthwise_conv",
    "detect_depthwise_conv",
    "detect_depthwise_conv_bn",
]

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect.utils import (
    is_consecutive_nodes,
    validate_bn_inputs,
)


def _get_conv_group_attr(node: NodeProto) -> int:
    """Extract group attribute from Conv node.

    :param node: Conv or ConvTranspose node
    :return: Group value (default 1)
    """
    for attr in node.attribute:
        if attr.name == "group":
            return int(attr.i)
    return 1


def _is_depthwise_conv(node: NodeProto, initializers: dict[str, TensorProto]) -> bool:
    """Check if a Conv node is depthwise convolution.

    Depthwise convolution: group == in_channels == out_channels
    This means each input channel has its own filter kernel.

    :param node: Conv node to check
    :param initializers: Model initializers (for weight shape)
    :return: True if depthwise convolution
    """
    if node.op_type not in {"Conv", "ConvTranspose"}:
        return False

    # Get group attribute
    group = _get_conv_group_attr(node)
    if group == 1:
        return False

    # Get weight tensor to check channels
    if len(node.input) < 2 or node.input[1] not in initializers:
        return False

    weight_tensor = initializers[node.input[1]]
    weight_shape = [int(d) for d in weight_tensor.dims]

    # Conv weight shape: [out_channels, in_channels/group, kH, kW]
    # For depthwise: group == in_channels, so in_channels/group == 1
    # And out_channels == in_channels == group
    if node.op_type == "Conv":
        out_channels = weight_shape[0]
        in_channels_per_group = weight_shape[1]

        # Depthwise condition: in_channels_per_group == 1 and group == out_channels
        return bool(in_channels_per_group == 1 and group == out_channels)
    # ConvTranspose
    # ConvTranspose weight shape: [in_channels, out_channels/group, kH, kW]
    in_channels = weight_shape[0]
    out_channels_per_group = weight_shape[1]

    # Depthwise condition: out_channels_per_group == 1 and group == in_channels
    return bool(out_channels_per_group == 1 and group == in_channels)


def detect_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
) -> list[dict]:
    """Detect standalone depthwise convolution operations.

    Depthwise convolution is a special case where:
    - group == in_channels == out_channels
    - Each input channel is convolved with its own filter

    This is commonly used in efficient architectures like MobileNet.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of depthwise convolution instances
    """
    instances = []

    for node in nodes:
        if _is_depthwise_conv(node, initializers):
            group = _get_conv_group_attr(node)
            weight_shape = list(initializers[node.input[1]].dims)

            instances.append(
                {
                    "node": node.name,
                    "op_type": node.op_type,
                    "group": group,
                    "weight": node.input[1],
                    "weight_shape": weight_shape,
                }
            )

    return instances


def detect_depthwise_conv_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
) -> list[dict]:
    """Detect Depthwise Conv + BatchNormalization fusion pattern.

    Pattern: DepthwiseConv -> BatchNormalization
    Can be fused by folding BN parameters into conv weights and biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 1):
        conv_node = nodes[i]
        bn_node = nodes[i + 1]

        # Check pattern: DepthwiseConv -> BatchNormalization
        if not _is_depthwise_conv(conv_node, initializers):
            continue

        if bn_node.op_type != "BatchNormalization":
            continue

        # Check consecutive connection
        if not is_consecutive_nodes(conv_node, bn_node, nodes):
            continue

        # Check if BN has constant parameters
        if not validate_bn_inputs(bn_node, initializers):
            continue

        group = _get_conv_group_attr(conv_node)

        instances.append(
            {
                "conv_node": conv_node.name,
                "bn_node": bn_node.name,
                "op_type": conv_node.op_type,
                "group": group,
                "conv_weight": conv_node.input[1],
                "bn_scale": bn_node.input[1],
                "bn_bias": bn_node.input[2],
                "bn_mean": bn_node.input[3],
                "bn_var": bn_node.input[4],
            }
        )

    return instances


def detect_bn_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + Depthwise Conv fusion pattern.

    Pattern: BatchNormalization -> DepthwiseConv
    Can be fused by folding BN parameters into conv weights.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 1):
        bn_node = nodes[i]
        conv_node = nodes[i + 1]

        # Check pattern: BatchNormalization -> DepthwiseConv
        if bn_node.op_type != "BatchNormalization":
            continue

        if not _is_depthwise_conv(conv_node, initializers):
            continue

        # Check consecutive connection
        if not is_consecutive_nodes(bn_node, conv_node, nodes):
            continue

        # Check if BN has constant parameters
        if not validate_bn_inputs(bn_node, initializers):
            continue

        group = _get_conv_group_attr(conv_node)

        instances.append(
            {
                "bn_node": bn_node.name,
                "conv_node": conv_node.name,
                "op_type": conv_node.op_type,
                "group": group,
                "bn_scale": bn_node.input[1],
                "bn_bias": bn_node.input[2],
                "bn_mean": bn_node.input[3],
                "bn_var": bn_node.input[4],
                "conv_weight": conv_node.input[1],
            }
        )

    return instances
