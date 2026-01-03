__docformat__ = "restructuredtext"
__all__ = [
    "_fuse_bn_gemm",
    "_fuse_bn_reshape_gemm",
    "_fuse_gemm_reshape_bn",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import (
    _get_batchnorm_params,
    _get_gemm_params,
    _is_only_next_node,
    compute_batchnorm_fusion_params,
)


def _fuse_gemm_reshape_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Fuse a Gemm, a Reshape, and a BatchNormalization node into a Gemm and a Reshape node."""
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
            _alpha, _beta, trans_a, trans_b, weight, bias = _get_gemm_params(
                gemm_node, initializers, remove_initializers=True
            )
            epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
                bn_node, initializers, remove_initializers=True
            )
            reshape_shape = (
                onnx.numpy_helper.to_array(initializers[reshape_node.input[1]]).astype(int).tolist()
            )
            if trans_a != 0:
                raise ValueError(
                    f"Gemm node {gemm_node.name} has unsupported transA={trans_a}. "
                    "Only transA=0 is supported for Gemm-Reshape-BN fusion."
                )
            num_features = bias.shape[0]
            weight = weight.T if trans_b == 1 else weight

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

            # Preserve dtype from weight tensor to avoid float32/float64 mismatch
            target_dtype = weight.dtype
            bn_weight, bn_bias = compute_batchnorm_fusion_params(
                epsilon, scale, bn_param_bias, mean, var, target_dtype
            )
            new_weight = (weight * bn_weight.reshape(1, -1, 1, 1)).astype(target_dtype, copy=False)
            new_bias = (bias + bn_bias.reshape(-1, 1, 1)).astype(target_dtype, copy=False)
            new_weight = new_weight.reshape((-1, num_features)).T
            new_bias = new_bias.reshape((num_features,))

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

            initializers[gemm_node.input[1]] = new_weight
            initializers[gemm_node.input[2]] = new_bias

            new_gemm_node = onnx.NodeProto()
            new_gemm_node.CopyFrom(gemm_node)
            new_gemm_node.ClearField("attribute")

            new_reshape_node = onnx.NodeProto()
            new_reshape_node.CopyFrom(reshape_node)
            new_reshape_node.ClearField("output")
            new_reshape_node.output.extend(bn_node.output)

            new_nodes.append(new_gemm_node)
            new_nodes.append(new_reshape_node)

        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    return new_nodes


def _fuse_bn_reshape_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Fuse a BatchNormalization, a Reshape, and a Gemm node into a Reshape and a Gemm node."""
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
            _alpha, _beta, trans_a, trans_b, weight, bias = _get_gemm_params(
                gemm_node, initializers, remove_initializers=True
            )
            epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
                bn_node, initializers, remove_initializers=True
            )
            # reshape_shape = onnx.numpy_helper.to_array(
            # initializers[reshape_node.input[1]])
            # reshape_shape = reshape_shape.tolist()
            if trans_a != 0:
                raise ValueError(
                    f"Gemm node {gemm_node.name} has unsupported transA={trans_a}. "
                    "Only transA=0 is supported for BN-Reshape-Gemm fusion."
                )
            weight = weight.T if trans_b == 1 else weight

            """
            IDEA
            BN:
            bn_weight (c,) = scale / np.sqrt(var + epsilon)
            bn_bias (c,) = bn_param_bias - (mean * weight)
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
            bn_weight_shape = gemm_shape[0] // bn_shape[0]
            weight = weight.reshape(-1, bn_weight_shape, gemm_shape[1])

            # Preserve dtype from weight tensor to avoid float32/float64 mismatch
            target_dtype = weight.dtype
            bn_weight, bn_bias = compute_batchnorm_fusion_params(
                epsilon, scale, bn_param_bias, mean, var, target_dtype
            )
            new_weight = (bn_weight.reshape(-1, 1, 1) * weight).astype(target_dtype, copy=False)
            new_bias = (bias + np.sum(bn_bias.reshape(-1, 1, 1) * weight, axis=(0, 1))).astype(
                target_dtype, copy=False
            )
            new_weight = new_weight.reshape(*gemm_shape).T

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

            initializers[new_weight_name] = new_weight
            initializers[new_bias_name] = new_bias

            new_reshape_node = onnx.NodeProto()
            new_reshape_node.CopyFrom(reshape_node)
            new_reshape_node.ClearField("input")
            new_reshape_node.input.extend([bn_node.input[0], reshape_node.input[1]])

            new_gemm_node = onnx.NodeProto()
            new_gemm_node.CopyFrom(gemm_node)
            new_gemm_node.ClearField("attribute")

            new_nodes.append(new_reshape_node)
            new_nodes.append(new_gemm_node)

        else:
            new_nodes.append(node)

        pre_pre_node = pre_node
        pre_node = node

    return new_nodes


def _fuse_bn_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Fuse a BatchNormalization and a Gemm node into a Gemm node."""
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
            _alpha, _beta, trans_a, trans_b, weight, bias = _get_gemm_params(
                gemm_node, initializers, remove_initializers=True
            )
            epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
                bn_node, initializers, remove_initializers=True
            )
            if trans_a != 0:
                raise ValueError(
                    f"Gemm node {gemm_node.name} has unsupported transA={trans_a}. "
                    "Only transA=0 is supported for BN-Gemm fusion."
                )
            weight = weight.T if trans_b == 1 else weight

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
            # Preserve dtype from weight tensor to avoid float32/float64 mismatch
            target_dtype = weight.dtype
            bn_weight, bn_bias = compute_batchnorm_fusion_params(
                epsilon, scale, bn_param_bias, mean, var, target_dtype
            )
            new_weight = (bn_weight.reshape(-1, 1) * weight).astype(target_dtype, copy=False)
            new_bias = (bias + np.sum(bn_bias.reshape(-1, 1) * weight, axis=0)).astype(
                target_dtype, copy=False
            )

            new_weight_name = gemm_node.input[1]
            new_bias_name = gemm_node.input[2]

            new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
            new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)

            initializers[gemm_node.input[1]] = new_weight
            initializers[gemm_node.input[2]] = new_bias

            new_node = onnx.NodeProto()
            new_node.CopyFrom(gemm_node)
            new_node.ClearField("input")
            new_node.ClearField("attribute")
            new_node.input.extend([bn_node.input[0], gemm_node.input[1], gemm_node.input[2]])

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
