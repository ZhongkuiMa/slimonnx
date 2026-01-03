__docformat__ = "restructuredtext"
__all__ = ["_fuse_conv_bn_or_bn_conv", "_fuse_conv_transpose_bn_or_bn_conv_transpose"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import (
    _get_batchnorm_params,
    _get_conv_params,
    _get_conv_transpose_params,
    _is_only_next_node,
    compute_batchnorm_fusion_params,
)


def _fuse_conv_bn_or_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
) -> list[NodeProto]:
    new_nodes = []
    pre_node = None
    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        if is_conv_bn and not (pre_node.op_type == "Conv" and node.op_type == "BatchNormalization"):
            pre_node = node
            continue

        if not is_conv_bn and not (
            pre_node.op_type == "BatchNormalization" and node.op_type == "Conv"
        ):
            pre_node = node
            continue

        # Pop the last two nodes
        new_nodes.pop()
        new_nodes.pop()

        if is_conv_bn:
            conv_node, bn_node = pre_node, node
        else:
            conv_node, bn_node = node, pre_node

        # Check if Conv has padding BEFORE calling _get_*_params to avoid modifying
        # initializers
        _, _, attrs = _get_conv_params(conv_node, initializers, remove_initializers=False)

        # Skip fusion if Conv has padding (fusion is incorrect with padding when
        # bn_bias != 0)
        if not is_conv_bn and any(p != 0 for p in attrs["pads"]):
            # Restore the nodes and skip fusion
            new_nodes.append(pre_node)
            new_nodes.append(node)
            pre_node = node
            continue

        # Now get params with get_from_initializers=True
        epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
            bn_node, initializers, remove_initializers=True
        )
        weight, bias, attrs = _get_conv_params(conv_node, initializers, remove_initializers=True)

        # Preserve dtype from weight tensor to avoid float32/float64 mismatch
        target_dtype = weight.dtype
        bn_weight, bn_bias = compute_batchnorm_fusion_params(
            epsilon, scale, bn_param_bias, mean, var, target_dtype
        )

        # If the bias is None, we have create a zero tensor in the above functions.
        if is_conv_bn:
            new_weight = (weight * bn_weight.reshape(-1, 1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (bias * bn_weight + bn_bias).astype(target_dtype, copy=False)
        else:
            new_weight = (weight * bn_weight.reshape(1, -1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (
                bias + np.sum(weight * bn_bias.reshape(1, -1, 1, 1), axis=(1, 2, 3))
            ).astype(target_dtype, copy=False)

        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[1] + "_bias"
        new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight
        initializers[new_bias_name] = new_bias

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


def _fuse_conv_transpose_bn_or_bn_conv_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_transpose_bn: bool = True,
) -> list[NodeProto]:
    new_nodes = []
    pre_node = None
    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        if is_conv_transpose_bn and not (
            pre_node.op_type == "ConvTranspose" and node.op_type == "BatchNormalization"
        ):
            pre_node = node
            continue

        if not is_conv_transpose_bn and not (
            pre_node.op_type == "BatchNormalization" and node.op_type == "ConvTranspose"
        ):
            pre_node = node
            continue

        # Pop the last two nodes
        new_nodes.pop()
        new_nodes.pop()

        if is_conv_transpose_bn:
            conv_node, bn_node = pre_node, node
        else:
            conv_node, bn_node = node, pre_node

        epsilon, scale, b, mean, var = _get_batchnorm_params(
            bn_node, initializers, remove_initializers=True
        )
        weight, bias, _attrs = _get_conv_transpose_params(
            conv_node, initializers, remove_initializers=True
        )

        # Preserve dtype from weight tensor to avoid float32/float64 mismatch
        target_dtype = weight.dtype
        bn_weight, bn_bias = compute_batchnorm_fusion_params(
            epsilon, scale, b, mean, var, target_dtype
        )

        if is_conv_transpose_bn:
            new_weight = (weight * bn_weight.reshape(1, -1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (bias * bn_weight + bn_bias).astype(target_dtype, copy=False)
        else:
            new_weight = (weight * bn_weight.reshape(-1, 1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (
                bias + np.sum(weight * bn_bias.reshape(-1, 1, 1, 1), axis=(0, 2, 3))
            ).astype(target_dtype, copy=False)

        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[1] + "_bias"
        new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight
        initializers[new_bias_name] = new_bias

        new_node = onnx.NodeProto()
        new_node.CopyFrom(conv_node)
        new_node.ClearField("input")
        new_node.ClearField("output")

        if is_conv_transpose_bn:
            new_node.input.extend([conv_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(bn_node.output)
        else:
            new_node.input.extend([bn_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(conv_node.output)

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
