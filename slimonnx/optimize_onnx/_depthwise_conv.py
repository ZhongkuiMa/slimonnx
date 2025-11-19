"""Depthwise convolution fusion optimizations."""

__docformat__ = "restructuredtext"
__all__ = [
    "_fuse_depthwise_conv_bn_or_bn_depthwise_conv",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from ._utils import (
    _is_only_next_node,
    _get_batchnorm_params,
    _get_conv_params,
    compute_batchnorm_fusion_params,
)


def _get_conv_group_attr(node: NodeProto) -> int:
    """Extract group attribute from Conv node.

    :param node: Conv node
    :return: Group value (default 1)
    """
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
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
    weight_shape = list(weight_tensor.dims)

    # Conv weight shape: [out_channels, in_channels/group, kH, kW]
    out_channels = weight_shape[0]
    in_channels_per_group = weight_shape[1]

    # Depthwise condition
    return in_channels_per_group == 1 and group == out_channels


def _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
    verbose: bool = False,
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
    count = 0
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
                _is_depthwise_conv(pre_node, initializers)
                and node.op_type == "BatchNormalization"
            )
        else:
            is_pattern = (
                pre_node.op_type == "BatchNormalization"
                and _is_depthwise_conv(node, initializers)
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
        epsilon, scale, b, mean, var = _get_batchnorm_params(
            bn_node, initializers, True
        )

        # Get Conv parameters
        weight, bias, attrs = _get_conv_params(conv_node, initializers, True)

        # Compute BN fusion parameters
        bn_weight, bn_bias = compute_batchnorm_fusion_params(
            epsilon, scale, b, mean, var
        )

        # Fuse parameters
        # For depthwise conv: weight shape is [C, 1, kH, kW]
        # BN parameters are [C]
        if is_conv_bn:
            # DepthwiseConv + BN
            new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
            new_bias = bias * bn_weight + bn_bias
        else:
            # BN + DepthwiseConv
            # For depthwise, each channel is independent
            new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
            new_bias = bias + bn_bias

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
        if is_conv_bn:
            inputs = [conv_node.input[0], new_weight_name, new_bias_name]
            outputs = bn_node.output
        else:
            inputs = [bn_node.input[0], new_weight_name, new_bias_name]
            outputs = conv_node.output

        new_node = onnx.helper.make_node(
            op_type="Conv",
            inputs=inputs,
            outputs=outputs,
            name=conv_node.name,
            kernel_shape=attrs["kernel_shape"],
            pads=attrs["pads"],
            strides=attrs["strides"],
            dilations=attrs["dilations"],
            group=attrs["group"],
            auto_pad=attrs["auto_pad"],
        )

        count += 1
        new_nodes.append(new_node)
        pre_node = node

    if verbose:
        direction = "DepthwiseConv-BN" if is_conv_bn else "BN-DepthwiseConv"
        print(f"Fused {count} {direction}.")

    return new_nodes
