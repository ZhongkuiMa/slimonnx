"""Depthwise convolution fusion optimizations."""

__docformat__ = "restructuredtext"
__all__ = [
    "_fuse_depthwise_conv_bn_or_bn_depthwise_conv",
]

import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import (
    _get_batchnorm_params,
    _get_conv_params,
    _is_only_next_node,
    compute_batchnorm_fusion_params,
)


def _get_conv_group_attr(node: NodeProto) -> int:
    """Extract group attribute from Conv node.

    :param node: Conv node
    :return: Group value (default 1)
    """
    for attr in node.attribute:
        if attr.name == "group":
            return int(attr.i)
    return 1


def _is_depthwise_conv(node: NodeProto, initializers: dict[str, TensorProto]) -> bool:
    """Check if a Conv node is depthwise convolution.

    Depthwise convolution: group == in_channels == out_channels
    Conv weight shape: [out_channels, in_channels/group, kH, kW]
    For depthwise: in_channels/group == 1 and group == out_channels

    :param node: Conv node to check
    :param initializers: Model initializers
    :return: True if depthwise convolution
    """
    if node.op_type != "Conv":
        return False

    group = _get_conv_group_attr(node)
    if group == 1:
        return False

    if len(node.input) < 2 or node.input[1] not in initializers:
        return False

    weight_tensor = initializers[node.input[1]]
    weight_shape = [int(d) for d in weight_tensor.dims]

    # Conv weight shape: [out_channels, in_channels/group, kH, kW]
    out_channels = weight_shape[0]
    in_channels_per_group = weight_shape[1]

    # Depthwise condition
    return bool(in_channels_per_group == 1 and group == out_channels)


def _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
) -> list[NodeProto]:
    """Fuse depthwise Conv + BatchNormalization or BatchNormalization + depthwise Conv.

    For depthwise convolution with group=C (channels):
    - Conv weight shape: [C, 1, kH, kW]
    - Each channel has its own 1x1 filter

    Fusion formula for DepthwiseConv+BN:
    - new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
    - new_bias = bias * bn_weight + bn_bias

    Fusion formula for BN+DepthwiseConv:
    - new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
    - new_bias = bias + bn_bias (simplified due to depthwise structure)

    :param nodes: List of nodes
    :param initializers: Dictionary of initializers
    :param is_conv_bn: True for Conv+BN, False for BN+Conv
    :param verbose: Print progress
    :return: Optimized nodes
    """
    new_nodes = []
    pre_node = None

    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        # Check pattern: DepthwiseConv + BN or BN + DepthwiseConv
        if is_conv_bn:
            is_pattern = (
                _is_depthwise_conv(pre_node, initializers) and node.op_type == "BatchNormalization"
            )
        else:
            is_pattern = pre_node.op_type == "BatchNormalization" and _is_depthwise_conv(
                node, initializers
            )

        if not is_pattern:
            pre_node = node
            continue

        # Pop the last two nodes
        new_nodes.pop()
        new_nodes.pop()

        if is_conv_bn:
            conv_node, bn_node = pre_node, node
        else:
            conv_node, bn_node = node, pre_node

        # Get BatchNorm parameters
        epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
            bn_node, initializers, remove_initializers=True
        )

        # Get Conv parameters
        weight, bias, _attrs = _get_conv_params(conv_node, initializers, remove_initializers=True)

        # Preserve dtype from weight tensor to avoid float32/float64 mismatch
        target_dtype = weight.dtype
        bn_weight, bn_bias = compute_batchnorm_fusion_params(
            epsilon, scale, bn_param_bias, mean, var, target_dtype
        )

        # Fuse parameters
        # For depthwise conv: weight shape is [C, 1, kH, kW]
        # BN parameters are [C]
        if is_conv_bn:
            # DepthwiseConv + BN
            new_weight = (weight * bn_weight.reshape(-1, 1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (bias * bn_weight + bn_bias).astype(target_dtype, copy=False)
        else:
            # BN + DepthwiseConv
            # For depthwise, each channel is independent
            new_weight = (weight * bn_weight.reshape(-1, 1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (bias + bn_bias).astype(target_dtype, copy=False)

        # Update initializers
        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[1] + "_bias"

        new_weight_tensor = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias_tensor = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight_tensor
        initializers[new_bias_name] = new_bias_tensor

        # Create fused node
        new_node = onnx.NodeProto()
        new_node.CopyFrom(conv_node)
        new_node.ClearField("input")
        new_node.ClearField("output")

        if is_conv_bn:
            new_node.input.extend([conv_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(bn_node.output)
        else:
            new_node.input.extend([bn_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(conv_node.output)

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
