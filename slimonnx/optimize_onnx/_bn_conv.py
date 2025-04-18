__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_conv_bn_or_bn_conv", "_fuse_convtranspose_bn"]

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

        new_node = node

        if (
            pre_node is not None
            and (
                (node.op_type == "Conv" and pre_node.op_type == "BatchNormalization")
                or (node.op_type == "BatchNormalization" and pre_node.op_type == "Conv")
            )
            and _in_single_path(pre_node, node, nodes)
        ):

            new_nodes.pop()
            if is_conv_bn:
                conv_node, bn_node = pre_node, node
            else:
                conv_node, bn_node = node, pre_node

            data_type = initializers[conv_node.input[1]].data_type
            epsilon, scale, b, mean, var = _get_batchnorm_params(bn_node, initializers)
            kernel_shape, pads, strides, dilations, group, auto_pad, weight, bias = (
                _get_conv_params(conv_node, initializers)
            )

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

            weight_name = conv_node.input[1]
            if len(conv_node.input) > 2:
                bias_name = conv_node.input[2]
            else:
                bias_name = conv_node.name + "_bias"

            new_weight = onnx.helper.make_tensor(
                name=weight_name,
                data_type=data_type,
                dims=new_weight.shape,
                vals=new_weight.flatten().tolist(),
            )
            new_bias = onnx.helper.make_tensor(
                name=bias_name,
                data_type=data_type,
                dims=new_bias.shape,
                vals=new_bias.flatten().tolist(),
            )

            initializers[weight_name] = new_weight
            initializers[bias_name] = new_bias

            if is_conv_bn:
                inputs = [conv_node.input[0], weight_name, bias_name]
                outputs = bn_node.output
            else:
                inputs = [bn_node.input[0], weight_name, bias_name]
                outputs = conv_node.output

            new_node = onnx.helper.make_node(
                op_type="Conv",
                inputs=inputs,
                outputs=outputs,
                name=conv_node.name,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                dilations=dilations,
                group=group,
                auto_pad=auto_pad,
            )
            count += 1

        new_nodes.append(new_node)
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} Conv-BN or BN-Conv.")

    return new_nodes


def _fuse_convtranspose_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    pre_node = None
    for node in nodes:

        new_node = node

        if (
            node.op_type == "BatchNormalization"
            and pre_node is not None
            and pre_node.op_type == "ConvTranspose"
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()
            conv_node, bn_node = pre_node, node
            data_type = initializers[conv_node.input[1]].data_type
            epsilon, scale, b, mean, var = _get_batchnorm_params(bn_node, initializers)
            (kernel_shape, pads, strides, dilations, group, auto_pad, weight, bias) = (
                _get_conv_params(conv_node, initializers)
            )

            bn_weight = scale / np.sqrt(var + epsilon)
            bn_bias = b - mean * bn_weight
            new_weight = weight * bn_weight.reshape(-1, 1, 1, 1)
            new_bias = bias * bn_weight + bn_bias

            new_weight = onnx.helper.make_tensor(
                name=conv_node.input[1],
                data_type=data_type,
                dims=new_weight.shape,
                vals=new_weight.flatten().tolist(),
            )
            new_bias = onnx.helper.make_tensor(
                name=conv_node.input[2],
                data_type=data_type,
                dims=new_bias.shape,
                vals=new_bias.flatten().tolist(),
            )

            initializers[conv_node.input[1]] = new_weight
            initializers[conv_node.input[2]] = new_bias

            new_node = onnx.helper.make_node(
                op_type="ConvTranspose",
                inputs=conv_node.input,
                outputs=bn_node.output,
                name=conv_node.name,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                dilations=dilations,
                group=group,
                auto_pad=auto_pad,
            )

            count += 1

        new_nodes.append(new_node)
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} ConvTranspose-BN.")

    return new_nodes
