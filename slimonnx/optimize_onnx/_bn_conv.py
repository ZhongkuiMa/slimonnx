__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_conv_bn_or_bn_conv", "_fuse_convtranspose_bn_or_bn_convtranspose"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ._utils import *


def _fuse_conv_bn_or_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    pre_node = None
    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        if is_conv_bn and not (
            (pre_node.op_type == "Conv" and node.op_type == "BatchNormalization")
        ):
            pre_node = node
            continue

        if not is_conv_bn and not (
            (pre_node.op_type == "BatchNormalization" and node.op_type == "Conv")
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

        epsilon, scale, b, mean, var = _get_batchnorm_params(
            bn_node, initializers, True
        )
        weight, bias, attrs = _get_conv_params(conv_node, initializers, True)

        bn_weight = scale / np.sqrt(var + epsilon)
        bn_bias = b - mean * bn_weight

        # If the bias is None, we have create a zero tensor in the above functions.
        if is_conv_bn:
            new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
            new_bias = bias * bn_weight + bn_bias
        else:
            new_weight = weight * bn_weight.reshape(1, -1, 1, 1)
            new_bias = bias + (
                np.sum(weight * bn_bias.reshape(1, -1, 1, 1), axis=(1, 2, 3))
            )

        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[2] + "_bias"
        new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight
        initializers[new_bias_name] = new_bias

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

    if utils.VERBOSE:
        print(f"Fused {count} Conv-BN or BN-Conv.")

    return new_nodes


def _fuse_convtranspose_bn_or_bn_convtranspose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_convtranspose_bn: bool = True,
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    pre_node = None
    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        if is_convtranspose_bn and not (
            (
                pre_node.op_type == "ConvTranspose"
                and node.op_type == "BatchNormalization"
            )
        ):
            pre_node = node
            continue

        if not is_convtranspose_bn and not (
            (
                pre_node.op_type == "BatchNormalization"
                and node.op_type == "ConvTranspose"
            )
        ):
            pre_node = node
            continue

        # Pop the last two nodes
        new_nodes.pop()
        new_nodes.pop()

        if is_convtranspose_bn:
            conv_node, bn_node = pre_node, node
        else:
            conv_node, bn_node = node, pre_node

        epsilon, scale, b, mean, var = _get_batchnorm_params(
            bn_node, initializers, True
        )
        weight, bias, attrs = _get_convtranspose_params(conv_node, initializers, True)

        bn_weight = scale / np.sqrt(var + epsilon)
        bn_bias = b - mean * bn_weight

        if is_convtranspose_bn:
            new_weight = weight * bn_weight.reshape(1, -1, 1, 1)
            new_bias = bias * bn_weight + bn_bias
        else:
            new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
            new_bias = bias + (
                np.sum(weight * bn_bias.reshape(-1, 1, 1, 1), axis=(0, 2, 3))
            )

        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[2] + "_bias"
        new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight
        initializers[new_bias_name] = new_bias

        if is_convtranspose_bn:
            inputs = [conv_node.input[0], new_weight_name, new_bias_name]
            outputs = bn_node.output
        else:
            inputs = [bn_node.input[0], new_weight_name, new_bias_name]
            outputs = conv_node.output

        new_node = onnx.helper.make_node(
            op_type="ConvTranspose",
            inputs=inputs,
            outputs=outputs,
            name=conv_node.name,
            kernel_shape=attrs["kernel_shape"],
            pads=attrs["pads"],
            strides=attrs["strides"],
            dilations=attrs["dilations"],
            output_shape=attrs["output_shape"],
            output_padding=attrs["output_padding"],
            group=attrs["group"],
            auto_pad=attrs["auto_pad"],
        )

        count += 1

        new_nodes.append(new_node)
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} ConvTranspose-BN.")

    return new_nodes
