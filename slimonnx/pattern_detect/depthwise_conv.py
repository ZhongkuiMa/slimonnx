"""Depthwise convolution pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_depthwise_conv",
    "detect_depthwise_conv_bn",
    "detect_bn_depthwise_conv",
]

from onnx import NodeProto, TensorProto


def _get_conv_group_attr(node: NodeProto) -> int:
    """Extract group attribute from Conv node.

    :param node: Conv or ConvTranspose node
    :return: Group value (default 1)
    """
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
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
    weight_shape = list(weight_tensor.dims)

    # Conv weight shape: [out_channels, in_channels/group, kH, kW]
    # For depthwise: group == in_channels, so in_channels/group == 1
    # And out_channels == in_channels == group
    if node.op_type == "Conv":
        out_channels = weight_shape[0]
        in_channels_per_group = weight_shape[1]

        # Depthwise condition: in_channels_per_group == 1 and group == out_channels
        return in_channels_per_group == 1 and group == out_channels
    else:  # ConvTranspose
        # ConvTranspose weight shape: [in_channels, out_channels/group, kH, kW]
        in_channels = weight_shape[0]
        out_channels_per_group = weight_shape[1]

        # Depthwise condition: out_channels_per_group == 1 and group == in_channels
        return out_channels_per_group == 1 and group == in_channels


def _is_consecutive_nodes(
    first_node: NodeProto, second_node: NodeProto, nodes: list[NodeProto]
) -> bool:
    """Check if second_node immediately follows first_node with no other consumers.

    :param first_node: First node
    :param second_node: Second node
    :param nodes: All nodes in graph
    :return: True if consecutive with single consumer
    """
    if not first_node.output or not second_node.input:
        return False

    if first_node.output[0] != second_node.input[0]:
        return False

    first_output = first_node.output[0]
    consumer_count = sum(1 for node in nodes if first_output in node.input)
    return consumer_count == 1


def detect_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
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
    data_shapes: dict[str, list[int]] | None = None,
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
        if not _is_consecutive_nodes(conv_node, bn_node, nodes):
            continue

        # Check if BN has constant parameters
        if len(bn_node.input) < 5:
            continue

        bn_params_available = all(inp in initializers for inp in bn_node.input[1:5])
        if not bn_params_available:
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
    data_shapes: dict[str, list[int]] | None = None,
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
        if not _is_consecutive_nodes(bn_node, conv_node, nodes):
            continue

        # Check if BN has constant parameters
        if len(bn_node.input) < 5:
            continue

        bn_params_available = all(inp in initializers for inp in bn_node.input[1:5])
        if not bn_params_available:
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
