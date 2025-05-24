__docformat__ = ["restructuredtext"]
__all__ = [
    "_fuse_gemm_reshape_bn",
    "_fuse_bn_reshape_gemm",
    "_fuse_bn_gemm",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ._utils import *


def _fuse_gemm_reshape_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """
    Fuse a Gemm, a Reshape, and a BatchNormalization node into a Gemm and a Reshape
    node.
    """
    count = 0

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
            and _is_only_next_node(pre_pre_node, pre_node, nodes)
            and _is_only_next_node(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()

            bn_node, reshape_node, gemm_node = node, pre_node, pre_pre_node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers, True
            )
            epsilon, scale, b, mean, var = _get_batchnorm_params(
                bn_node, initializers, True
            )
            reshape_shape = (
                onnx.numpy_helper.to_array(initializers[reshape_node.input[1]])
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

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

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

            count += 1

        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} Gemm-Reshape-BN.")

    return new_nodes


def _fuse_bn_reshape_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """
    Fuse a BatchNormalization, a Reshape, and a Gemm node into a Reshape and a Gemm
    node.
    """
    count = 0
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
            and _is_only_next_node(pre_pre_node, pre_node, nodes)
            and _is_only_next_node(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()
            bn_node, reshape_node, gemm_node = pre_pre_node, pre_node, node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers, True
            )
            epsilon, scale, b, mean, var = _get_batchnorm_params(
                bn_node, initializers, True
            )
            # reshape_shape = onnx.numpy_helper.to_array(
            # initializers[reshape_node.input[1]])
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

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

            initializers[new_weight_name] = new_weight
            initializers[new_bias_name] = new_bias

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

            count += 1

        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} BN-Reshape-Gemm.")

    return new_nodes


def _fuse_bn_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """
    Fuse a BatchNormalization and a Gemm node into a Gemm node.
    """
    count = 0

    new_nodes = []
    pre_node = None
    for node in nodes:
        new_node = node
        if (
            node.op_type == "Gemm"
            and pre_node is not None
            and pre_node.op_type == "BatchNormalization"
            and _is_only_next_node(pre_node, node, nodes)
        ):
            new_nodes.pop()

            gemm_node, bn_node = node, pre_node
            data_type = initializers[gemm_node.input[1]].data_type
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers, True
            )
            epsilon, scale, b, mean, var = _get_batchnorm_params(
                bn_node, initializers, True
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

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

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

            count += 1

        new_nodes.append(new_node)
        pre_node = node

    if utils.VERBOSE:
        print(f"Fused {count} BN-Gemm.")

    return new_nodes
