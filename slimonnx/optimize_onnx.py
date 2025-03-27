__docformat__ = "restructuredtext"
__all__ = ["optimize_onnx"]

import math
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from slimonnx.get_attrs import get_onnx_node_attrs
from slimonnx.infer_shape import infer_onnx_shape
from slimonnx.utils import *


def _in_single_path(
    pre_node: onnx.NodeProto, node: onnx.NodeProto, nodes: list[onnx.NodeProto]
) -> bool:
    """
    Check the pre_node and node are in a single path. Because if there are multiple
    paths, we cannot fuse the nodes to avoid changing the computation graph.
    """
    pre_node_name = pre_node.output[0]
    for n in nodes:
        if pre_node_name in n.input and n != node:
            return False
    return True


def _get_batch_normalization_params(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    remove_initializers: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the parameters of a BatchNormalization node.
    """
    epsilon = get_onnx_node_attrs(node)["epsilon"]
    scale = numpy_helper.to_array(initializers[node.input[1]])
    b = numpy_helper.to_array(initializers[node.input[2]])
    mean = numpy_helper.to_array(initializers[node.input[3]])
    var = numpy_helper.to_array(initializers[node.input[4]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]
        del initializers[node.input[3]]
        del initializers[node.input[4]]

    return epsilon, scale, b, mean, var


def _get_gemm_params(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
    remove_initializers: bool = True,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    """
    Get the parameters of a Gemm node.
    """
    attrs = get_onnx_node_attrs(node)
    alpha = attrs["alpha"]
    beta = attrs["beta"]
    transA = attrs["transA"]
    transB = attrs["transB"]
    weight = numpy_helper.to_array(initializers[node.input[1]])
    bias = numpy_helper.to_array(initializers[node.input[2]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]

    return alpha, beta, transA, transB, weight, bias


def _get_conv_params(
    node: onnx.NodeProto,
    initializers: dict[str, onnx.TensorProto],
):
    """
    Get the parameters of a Conv or ConvTranspose node.
    """
    attrs = get_onnx_node_attrs(node)
    kernel_shape = attrs["kernel_shape"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    dilations = attrs["dilations"]
    group = attrs["group"]
    auto_pad = attrs["auto_pad"]

    weight = numpy_helper.to_array(initializers[node.input[1]])
    del initializers[node.input[1]]

    if len(node.input) == 2:  # No bias
        bias = np.zeros(weight.shape[0])
    else:
        bias = numpy_helper.to_array(initializers[node.input[2]])
        del initializers[node.input[2]]

    return kernel_shape, pads, strides, dilations, group, auto_pad, weight, bias


def _fuse_matmul_add(
    nodes: list[onnx.NodeProto], initializers: dict[str, onnx.TensorProto]
) -> list[onnx.NodeProto]:
    """
    Fuse a MatMul and an Add node into a single Gemm node.
    """
    new_nodes = []
    pre_node = None
    for node in nodes:
        new_node = node
        if (
            node.op_type == "Add"
            and pre_node is not None
            and (node.input[0] in initializers or node.input[1] in initializers)
            and pre_node.op_type == "MatMul"
            and (pre_node.input[0] in initializers or pre_node.input[1] in initializers)
            and _in_single_path(pre_node, node, nodes)
        ):
            matmul_node, add_node = pre_node, node
            input_name, weight_name, transB = (
                (matmul_node.input[0], matmul_node.input[1], 0)
                if matmul_node.input[1] in initializers
                else (matmul_node.input[1], matmul_node.input[0], 1)
            )
            bias_name = (
                add_node.input[1]
                if add_node.input[0] == matmul_node.output[0]
                else add_node.input[0]
            )

            can_fuse = True
            weight_dim = len(initializers[weight_name].dims)
            if weight_dim != 2:
                can_fuse = False
            bias_dim = len(initializers[bias_name].dims)
            if bias_dim != 1:
                can_fuse = False

            if can_fuse:
                new_nodes.pop()
                inputs = (
                    [input_name, weight_name, bias_name]
                    if transB == 0
                    else [weight_name, input_name, bias_name]
                )
                new_node = onnx.helper.make_node(
                    op_type="Gemm", inputs=inputs, outputs=add_node.output
                )

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes


def _fuse_gemm_reshape_bn(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """
    Fuse a Gemm, a Reshape, and a BatchNormalization node into a Gemm and a Reshape
    node.
    """
    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        if (
            node.op_type == "BatchNormalization"
            and pre_pre_node is not None
            and pre_node is not None
            and pre_pre_node.op_type == "Gemm"
            and pre_node.op_type == "Reshape"
            and _in_single_path(pre_pre_node, pre_node, nodes)
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()

            bn_node, reshape_node, gemm_node = node, pre_node, pre_pre_node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers
            )
            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )
            reshape_shape = (
                numpy_helper.to_array(initializers[reshape_node.input[1]])
                .astype(int)
                .tolist()
            )
            assert transA == 0
            n = bias.shape[0]
            weight = weight.T if transB == 1 else weight

            """
            IDEA
            GEMM: (M, K) @ (K, N) + (N,) => (M, N)
            Reshape: (M, N) => (M, c, h, w)
            BN: 
            bn_weight: (c,) = scale / np.sqrt(var + epsilon)
            bn_bias: (c,) = b - (mean * bn_weight)
            
            We need BN to be
                weight (c,) => (1, c, 1, 1)
                bias (c,) => (c, 1, 1)
            We need GEMM to be
                weight (K, N) => (K, c, h, w)
                bias (N,) => (c, h, w)
            Then, we have
                weight <- (K, c, h, w) * (1, c, 1, 1)
                bias <- (c, h, w) * (c, 1, 1)
            """
            weight = weight.reshape(-1, *reshape_shape[1:])
            bias = bias.reshape(*reshape_shape[1:])
            bn_weight = scale / np.sqrt(var + epsilon)
            bn_bias = b - mean * bn_weight
            new_weight = weight * bn_weight.reshape(1, -1, 1, 1)
            new_bias = bias + bn_bias.reshape(-1, 1, 1)
            new_weight = new_weight.reshape((-1, n)).T
            new_bias = new_bias.reshape((n,))

            new_weight = onnx.helper.make_tensor(
                name=gemm_node.input[1],
                data_type=data_type,
                dims=new_weight.shape,
                vals=new_weight.flatten().tolist(),
            )
            new_bias = onnx.helper.make_tensor(
                name=gemm_node.input[2],
                data_type=data_type,
                dims=new_bias.shape,
                vals=new_bias.flatten().tolist(),
            )
            initializers[gemm_node.input[1]] = new_weight
            initializers[gemm_node.input[2]] = new_bias

            new_gemm_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=gemm_node.input,
                outputs=gemm_node.output,
                name=gemm_node.name,
                alpha=alpha,
                beta=beta,
                transA=transA,
                transB=transB,
            )

            new_reshape_node = onnx.helper.make_node(
                op_type="Reshape",
                inputs=reshape_node.input,
                outputs=bn_node.output,
                name=reshape_node.name,
            )

            new_nodes.append(new_gemm_node)
            new_nodes.append(new_reshape_node)
        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    return new_nodes


def _fuse_bn_reshape_gemm(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """
    Fuse a BatchNormalization, a Reshape, and a Gemm node into a Reshape and a Gemm
    node.
    """
    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        if (
            node.op_type == "Gemm"
            and pre_pre_node is not None
            and pre_node is not None
            and pre_pre_node.op_type == "BatchNormalization"
            and pre_node.op_type == "Reshape"
            and _in_single_path(pre_pre_node, pre_node, nodes)
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()
            bn_node, reshape_node, gemm_node = pre_pre_node, pre_node, node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers
            )
            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )
            # reshape_shape = numpy_helper.to_array(initializers[reshape_node.input[1]])
            # reshape_shape = reshape_shape.tolist()
            assert transA == 0
            weight = weight.T if transB == 1 else weight

            """
            IDEA
            BN: 
            bn_weight (c,) = scale / np.sqrt(var + epsilon)
            bn_bias (c,) = b - (mean * weight)
            Reshape: (M, c, h * w) => (M, K) (case (M, c, h, w) is similar)
            GEMM: (M, K) @ (K, N) + (N,) => (M, N)
            
            We need BN to be
                weight (c,) => (c, 1, 1)
                bias (c,) => (c, 1, 1)
            We need GEMM to be
                weight (K, N) => (c, h * w, N)
                bias (K,) => (N,)
            Then, we have
                weight <- (c, 1, 1) * (c, h * w, N)
                bias <- (N,) * ((c, 1, 1) * (c, h * w, N)).sum(axis=(0, 1))
            """
            bn_shape = scale.shape
            gemm_shape = weight.shape
            bn_weight = gemm_shape[0] // bn_shape[0]
            weight = weight.reshape(-1, bn_weight, gemm_shape[1])
            bn_weight = scale / np.sqrt(var + epsilon)
            bn_bias = b - mean * bn_weight
            new_weight = bn_weight.reshape(-1, 1, 1) * weight
            new_bias = bias + np.sum(bn_bias.reshape(-1, 1, 1) * weight, axis=(0, 1))

            new_weight = new_weight.reshape(*gemm_shape).T
            new_weight = onnx.helper.make_tensor(
                name=gemm_node.input[1],
                data_type=data_type,
                dims=new_weight.shape,
                vals=new_weight.flatten().tolist(),
            )
            new_bias = onnx.helper.make_tensor(
                name=gemm_node.input[2],
                data_type=data_type,
                dims=new_bias.shape,
                vals=new_bias.flatten().tolist(),
            )
            initializers[gemm_node.input[1]] = new_weight
            initializers[gemm_node.input[2]] = new_bias

            new_reshape_node = onnx.helper.make_node(
                op_type="Reshape",
                inputs=[bn_node.input[0], reshape_node.input[1]],
                outputs=reshape_node.output,
                name=reshape_node.name,
            )
            new_gemm_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=gemm_node.input,
                outputs=gemm_node.output,
                name=gemm_node.name,
                alpha=alpha,
                beta=beta,
                transA=transA,
                transB=transB,
            )

            new_nodes.append(new_reshape_node)
            new_nodes.append(new_gemm_node)
        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    return new_nodes


def _fuse_bn_gemm(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """
    Fuse a BatchNormalization and a Gemm node into a Gemm node.
    """
    new_nodes = []
    pre_node = None
    for node in nodes:
        new_node = node
        if (
            node.op_type == "Gemm"
            and pre_node is not None
            and pre_node.op_type == "BatchNormalization"
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()

            gemm_node, bn_node = node, pre_node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers
            )
            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )
            assert transA == 0
            weight = weight.T if transB == 1 else weight

            """
            IDEA
            BN: (K,)
            GEMM: (M, K) @ (K, N) + (N,) => (M, N)
            
            We need BN to be
                bn_weight (K,) = (K, 1)
                bn_bias (K,) = (K, 1)
            Then, we have
                weight (K, N) = (K, N) * (K, 1)
                bias (N,) = (N,) - ((K, 1) * (K, N)).sum(axis=0)
            """
            bn_weight = scale / np.sqrt(var + epsilon)
            bn_bias = b - mean * bn_weight
            new_weight = bn_weight.reshape(-1, 1) * weight
            new_bias = bias + np.sum(bn_bias.reshape(-1, 1) * weight, axis=0)

            new_weight = onnx.helper.make_tensor(
                name=gemm_node.input[1],
                data_type=data_type,
                dims=new_weight.shape,
                vals=new_weight.flatten().tolist(),
            )
            new_bias = onnx.helper.make_tensor(
                name=gemm_node.input[2],
                data_type=data_type,
                dims=new_bias.shape,
                vals=new_bias.flatten().tolist(),
            )
            initializers[gemm_node.input[1]] = new_weight
            initializers[gemm_node.input[2]] = new_bias

            new_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=[bn_node.input[0], gemm_node.input[1], gemm_node.input[2]],
                outputs=gemm_node.output,
                name=gemm_node.name,
                alpha=alpha,
                beta=beta,
            )

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes


def _fuse_gemm_gemm(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    new_nodes = []

    all_gemm_nodes = [node.output[0] for node in nodes if node.op_type == "Gemm"]
    is_next_node_gemm: dict[str, Any] = {node_name: [] for node_name in all_gemm_nodes}
    for node in nodes:
        is_gemm = node.op_type == "Gemm"
        for input_i in node.input:
            if input_i in all_gemm_nodes:
                is_next_node_gemm[input_i].append(is_gemm)
    for node_name, is_next_node_gemm_i in is_next_node_gemm.items():
        is_next_node_gemm[node_name] = (
            all(is_next_node_gemm_i) and len(is_next_node_gemm_i) > 0
        )

    gemm_node_names_to_fuse = [
        node_name
        for node_name, is_next_node_gemm in is_next_node_gemm.items()
        if is_next_node_gemm
    ]
    gemm_nodes_to_fuse = {
        node.output[0]: node
        for node in nodes
        if node.output[0] in gemm_node_names_to_fuse
    }

    for node in nodes:
        new_node = node
        if node.op_type == "Gemm":
            if node.output[0] in gemm_nodes_to_fuse:
                continue
            elif node.input[0] in gemm_nodes_to_fuse:
                # Fuse the current node with the previous node
                pre_node = gemm_nodes_to_fuse[node.input[0]]
                data_type = initializers[node.input[1]].data_type
                alpha1, beta1, transA1, transB1, weight1, bias1 = _get_gemm_params(
                    node, initializers
                )
                alpha2, beta2, transA2, transB2, weight2, bias2 = _get_gemm_params(
                    pre_node, initializers, remove_initializers=False
                )
                assert alpha1 == alpha2 == beta1 == beta2 == 1
                assert transA1 == transA2 == transB1 == transB2 == 0

                """
                IDEA
                   (X @ W_2 + b_2) @ W_1 + b_1
                => X @ (W_2 @ W_1) + (b_2 @ W_1 + b_1)
                """
                new_weight = weight2 @ weight1
                new_bias = bias2 @ weight1 + bias1

                new_weight = onnx.helper.make_tensor(
                    name=node.input[1],
                    data_type=data_type,
                    dims=new_weight.shape,
                    vals=new_weight.flatten().tolist(),
                )
                new_bias = onnx.helper.make_tensor(
                    name=node.input[2],
                    data_type=data_type,
                    dims=new_bias.shape,
                    vals=new_bias.flatten().tolist(),
                )
                initializers[node.input[1]] = new_weight
                initializers[node.input[2]] = new_bias

                new_node = onnx.helper.make_node(
                    op_type="Gemm",
                    inputs=[pre_node.input[0], node.input[1], node.input[2]],
                    outputs=node.output,
                    name=node.name,
                )
        new_nodes.append(new_node)

    # Remove the initializers of the fused nodes
    for node in gemm_nodes_to_fuse.values():
        del initializers[node.input[1]]
        del initializers[node.input[2]]

    return new_nodes


def _fuse_conv_bn_or_bn_conv(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
    is_conv_bn: bool = True,
) -> list[onnx.NodeProto]:
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
            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )
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

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes


def _fuse_convtranspose_bn(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    new_nodes = []
    pre_node = None
    for node in nodes:
        # print(node.op_type, node.input, node.output)
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
            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )
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

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes


def _fuse_transpose_batchnorm_transpose(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> list[onnx.NodeProto]:
    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        # print(node.op_type, node.input, node.output)
        new_node = node
        if (
            node.op_type == "Transpose"
            and pre_pre_node is not None
            and pre_node is not None
            and pre_pre_node.op_type == "Transpose"
            and pre_node.op_type == "BatchNormalization"
            and _in_single_path(pre_pre_node, pre_node, nodes)
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()
            tp_node1, bn_node, tp_node2 = pre_pre_node, pre_node, node
            data_type = initializers[bn_node.input[2]].data_type

            perm1 = get_onnx_node_attrs(tp_node1)["perm"]
            perm2 = get_onnx_node_attrs(tp_node2)["perm"]
            assert perm1 == [0, 2, 1]
            assert perm2 == [0, 2, 1]

            epsilon, scale, b, mean, var = _get_batch_normalization_params(
                bn_node, initializers
            )

            weight_name = bn_node.input[1] + "_gemm"
            bias_name = bn_node.input[2] + "_gemm"
            bn_weight = scale / np.sqrt(var + epsilon)
            weight = np.diag(bn_weight)
            bias = b - mean * bn_weight
            weight = onnx.helper.make_tensor(
                name=weight_name,
                data_type=data_type,
                dims=weight.shape,
                vals=weight.flatten().tolist(),
            )
            bias = onnx.helper.make_tensor(
                name=bias_name,
                data_type=data_type,
                dims=bias.shape,
                vals=bias.flatten().tolist(),
            )
            initializers[weight_name] = weight
            initializers[bias_name] = bias

            new_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=[tp_node1.input[0], weight_name, bias_name],
                outputs=tp_node2.output,
                name=bn_node.name + "_gemm",
            )

        new_nodes.append(new_node)
        pre_pre_node = pre_node
        pre_node = node

    return new_nodes


def _shape_to_initializer(
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
    shapes: dict[str, list[int]],
) -> list[onnx.NodeProto]:

    shape_nodes = {}
    shape_trace_values = {}
    # Iterate over value_info to print tensor names and their shapes
    for node in nodes:
        if node.op_type == "Shape":
            shape = shapes[node.input[0]]
            shape_nodes[node.output[0]] = shape
            shape_trace_values[node.output[0]] = shape
            # print("Get Shape node:", node.output[0], shape)
        else:
            for input_name in node.input:
                if input_name in shape_trace_values:
                    cur_data = shape_trace_values[input_name]
                    # print(f"TRACE shape={input_name} => {node.output[0]}")
                    # print(f"Current data: {shape_trace_values[input_name]}")
                    # print(node.output[0])

                    if node.op_type == "Gather":
                        indices = numpy_helper.to_array(initializers[node.input[1]])
                        new_shape = cur_data[indices]
                        shape_trace_values[node.output[0]] = new_shape
                        # print(f"Change data: {shape_trace_values[node.output[0]]}")

                    elif node.op_type == "Slice":
                        assert len(node.input) == 4, "Only consider 4 inputs"
                        starts = initializers[node.input[1]].int64_data[0]
                        ends = initializers[node.input[2]].int64_data[0]
                        # axes must be 0
                        shape_trace_values[node.output[0]] = cur_data[starts:ends]
                        # print(f"Change data: {shape_trace_values[node.output[0]]}")

                    elif node.op_type == "Reshape":
                        # Create an initializer for the shape and stop the trace.
                        ori_shape = shapes[node.input[0]]
                        inferred_shape = math.prod(ori_shape)
                        inferred_idx = -1
                        for idx, shape_i in enumerate(cur_data):
                            if shape_i == -1:
                                inferred_idx = idx
                                continue
                            inferred_shape = inferred_shape // shape_i
                        if inferred_idx != -1:
                            cur_data[inferred_idx] = inferred_shape

                        shape_name = node.input[1]
                        shape = onnx.helper.make_tensor(
                            name=shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(cur_data)],
                            vals=cur_data,
                        )
                        initializers[shape_name] = shape
                        # print(f"Finish trace {cur_data} of {shape_name}")

                    elif node.op_type == "Concat":
                        if isinstance(cur_data, list):
                            # This is a shape
                            new_shape = []
                            for input_i in node.input:
                                if input_i in shape_trace_values:
                                    temp = shape_trace_values[input_i]
                                    new_shape.extend(temp)
                                else:
                                    temp = numpy_helper.to_array(initializers[input_i])
                                    new_shape.append(temp[0])
                            shape_trace_values[node.output[0]] = new_shape
                            # print(f"Change data: {shape_trace_values[node.output[0]]}")
                        else:
                            assert isinstance(cur_data, tuple)
                            new_initializer = onnx.helper.make_tensor(
                                name=input_name,
                                data_type=cur_data[2],
                                dims=cur_data[1],
                                vals=cur_data[0],
                            )
                            initializers[input_name] = new_initializer
                            # print(f"Finish trace {cur_data} of {input_name}")

                    elif node.op_type == "Unsqueeze":
                        new_shape = cur_data
                        if isinstance(cur_data, int):
                            new_shape = [cur_data]
                        shape_trace_values[node.output[0]] = new_shape
                        # print(f"Change data: {shape_trace_values[node.output[0]]}")

                    elif node.op_type == "ConstantOfShape":
                        shape = cur_data
                        value = node.attribute[0]
                        data_type = value.t.data_type
                        value = numpy_helper.to_array(value.t)[0]
                        constant = (
                            np.full(shape, value, dtype=value.dtype),
                            shape,
                            data_type,
                        )

                        shape_trace_values[node.output[0]] = constant
                        # print(f"Change data: {shape_trace_values[node.output[0]]}")

                    elif node.op_type == "Add":
                        cur_value = cur_data[0]
                        another_value = numpy_helper.to_array(
                            initializers[node.input[1]]
                        )
                        new_value = cur_value + another_value
                        new_value = (new_value, *cur_data[1:])
                        shape_trace_values[node.output[0]] = new_value
                        # print(f"Change data: {shape_trace_values[node.output[0]]}")

                    else:
                        raise NotImplementedError(
                            f"Unknown op_type: {node.op_type} of {node.output[0]}"
                        )

                    # print()

    # Remove all shape nodes
    new_nodes = []
    for node in nodes:
        if node.output[0] not in shape_trace_values.keys():
            new_nodes.append(node)

    return new_nodes


def _simplify_names(
    input_nodes: list[onnx.ValueInfoProto],
    output_nodes: list[onnx.ValueInfoProto],
    nodes: list[onnx.NodeProto],
    initializers: dict[str, onnx.TensorProto],
) -> tuple[list[onnx.NodeProto], dict[str, onnx.TensorProto]]:
    """
    Simplify the names of the nodes and initializers.
    """
    node_output_names_mapping = {}

    # Change the node name of all nodes
    counter = 0
    for node in input_nodes:
        new_name = f"input_{counter}"
        node_output_names_mapping[node.name] = new_name  # For update input names
        node.name = new_name
        counter += 1

    for node in nodes:
        op_type = str(node.op_type)
        node.name = f"{op_type}_{counter}"
        counter += 1

    assert len(output_nodes) == 1
    for node in output_nodes:
        new_name = f"output_{counter}"
        node.name = new_name
        counter += 1

    # Change the node output name of all nodes
    for node in nodes:
        new_output_names = []
        for idx, output_name in enumerate(node.output):
            new_output_name = f"{node.name}_{idx}"
            new_output_names.append(new_output_name)
            node_output_names_mapping[output_name] = new_output_name
        # Change the original output names
        node.output[:] = new_output_names

    # Set the output name
    last_node = nodes[-1]
    last_node.output[:] = [output_nodes[0].name]

    # Change the input name of all nodes
    for node in nodes:
        new_input_names = []
        for input_name in node.input:
            if input_name in node_output_names_mapping:
                new_input_names.append(node_output_names_mapping[input_name])
            else:
                new_input_names.append(input_name)
        node.input[:] = new_input_names

    # Change the initializer name of all initializers
    # There maybe one initializer is not used by more than one node
    # So we number them dependently.
    counter = 0
    new_initializers = {}
    initializers_name_mapping = {}
    for name, initializer in initializers.items():
        new_name = f"Initializer_{counter}"
        new_initializers[new_name] = initializer
        initializer.name = new_name
        initializers_name_mapping[name] = new_name
        counter += 1

    for node in nodes:
        for idx, input_name in enumerate(node.input):
            if input_name in initializers_name_mapping:
                node.input[idx] = initializers_name_mapping[input_name]

    return nodes, new_initializers


def _reorder_by_strict_topological_order(
    nodes: list[onnx.NodeProto],
) -> list[onnx.NodeProto]:

    next_nodes_mapping = get_next_nodes_mapping(nodes)

    # Topological sort
    visited = {node.name: False for node in nodes}
    stack = []

    def _topological_sort(node_name: str):
        nonlocal visited
        nonlocal next_nodes_mapping
        nonlocal stack
        visited[node_name] = True
        for next_node_name in next_nodes_mapping[node_name]:
            if not visited[next_node_name]:
                _topological_sort(next_node_name)
        stack.append(node_name)

    for node in nodes:
        if not visited[node.name]:
            _topological_sort(node.name)
    stack.reverse()

    # Reorder the nodes
    name_node_mapping = {node.name: node for node in nodes}
    new_nodes = [name_node_mapping[node_name] for node_name in stack]

    return new_nodes


def optimize_onnx(
    model: onnx.ModelProto,
    fuse_matmul_add: bool = False,
    fuse_gemm_reshape_bn: bool = False,
    fuse_bn_reshape_gemm: bool = False,
    fuse_bn_gemm: bool = False,
    fuse_transpose_bn_transpose: bool = False,
    fuse_gemm_gemm: bool = False,
    fuse_conv_bn: bool = False,
    fuse_bn_conv: bool = False,
    fuse_transposedconv_bn: bool = False,
    shape_to_initializer: bool = False,
    simplify_node_name: bool = True,
    reorder_by_strict_topological_order: bool = True,
) -> onnx.ModelProto:
    clear_onnx_docstring(model)

    graph_name = model.graph.name + "_simplified"

    input_nodes = [
        onnx.helper.make_tensor_value_info(
            name=input_i.name,
            elem_type=input_i.type.tensor_type.elem_type,
            shape=[1] + [x.dim_value for x in input_i.type.tensor_type.shape.dim[1:]],
        )
        for input_i in model.graph.input
    ]

    output_nodes = [
        onnx.helper.make_tensor_value_info(
            name=output_i.name,
            elem_type=output_i.type.tensor_type.elem_type,
            shape=[1] + [x.dim_value for x in output_i.type.tensor_type.shape.dim[1:]],
        )
        for output_i in model.graph.output
    ]

    initializers = get_initializers(model)
    all_shapes = infer_onnx_shape(model, verbose=False)

    nodes = list(model.graph.node)

    if shape_to_initializer:
        nodes = _shape_to_initializer(nodes, initializers, all_shapes)
    if fuse_matmul_add:
        nodes = _fuse_matmul_add(nodes, initializers)
    if fuse_gemm_reshape_bn:
        nodes = _fuse_gemm_reshape_bn(nodes, initializers)
    if fuse_bn_reshape_gemm:
        nodes = _fuse_bn_reshape_gemm(nodes, initializers)
    if fuse_bn_gemm:
        nodes = _fuse_bn_gemm(nodes, initializers)
    if fuse_transpose_bn_transpose:
        nodes = _fuse_transpose_batchnorm_transpose(nodes, initializers)
    if fuse_gemm_gemm:
        nodes = _fuse_gemm_gemm(nodes, initializers)
    if fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
    if fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
    if fuse_transposedconv_bn:
        nodes = _fuse_convtranspose_bn(nodes, initializers)
    if reorder_by_strict_topological_order:
        # There maybe repeated named nodes, so we need to simplify the names first
        nodes, initializers = _simplify_names(
            input_nodes, output_nodes, nodes, initializers
        )
        nodes = _reorder_by_strict_topological_order(nodes)
    if simplify_node_name:
        # The name is ordered.
        nodes, initializers = _simplify_names(
            input_nodes, output_nodes, nodes, initializers
        )

    new_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            graph_name,
            input_nodes,
            output_nodes,
            list(initializers.values()),
        )
    )

    return new_model
