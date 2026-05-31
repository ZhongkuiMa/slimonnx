"""Shared utilities for ONNX graph optimization passes.

This module's public surface is ``compute_batchnorm_fusion_params``;
everything else is a subpackage-internal helper marked with the standard
single-underscore prefix and intentionally absent from ``__all__``.
"""

__docformat__ = "restructuredtext"
__all__ = ["compute_batchnorm_fusion_params"]

from typing import Any

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._onnx_attrs import get_onnx_attrs
from slimonnx.utils import has_single_consumer


def _is_only_next_node(
    pre_node: NodeProto,
    cur_node: NodeProto,
    nodes: list[NodeProto],
) -> bool:
    """Return ``True`` when ``cur_node`` is the only consumer of ``pre_node``.

    Linear fusion passes (Conv-BN, Gemm-BN, etc.) may rewrite the predecessor
    only when no other node depends on its output -- otherwise the fusion
    silently drops a forward edge from the computation graph.

    Delegates to the canonical ``slimonnx.utils.has_single_consumer``.

    :param pre_node: Candidate upstream node.

    :param cur_node: Candidate downstream node.

    :param nodes: Full graph node list, scanned for other consumers of
        ``pre_node.output[0]``.

    :return: ``True`` if no node other than ``cur_node`` reads
        ``pre_node.output[0]``.
    """
    return has_single_consumer(pre_node, cur_node, nodes)


def _get_batchnorm_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract BatchNormalization parameters from initializers.

    :param node: BatchNormalization NodeProto.

    :param initializers: Initializer map keyed by tensor name. When
        ``remove_initializers`` is true the four BN parameter initializers
        are popped from the map.

    :param remove_initializers: Whether to pop the consumed initializers.

    :return: Tuple ``(epsilon, scale, bias, mean, var)`` -- the BN affine
        parameters and the running statistics needed by the fusion math.
    """
    attrs = get_onnx_attrs(node, initializers)
    epsilon = attrs["epsilon"]
    scale = onnx.numpy_helper.to_array(initializers[node.input[1]])
    bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
    mean = onnx.numpy_helper.to_array(initializers[node.input[3]])
    var = onnx.numpy_helper.to_array(initializers[node.input[4]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]
        del initializers[node.input[3]]
        del initializers[node.input[4]]

    return epsilon, scale, bias, mean, var


def _get_gemm_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    """Extract Gemm scalar attrs and weight/bias initializers.

    :param node: Gemm NodeProto.

    :param initializers: Initializer map keyed by tensor name. When
        ``remove_initializers`` is true the weight and bias initializers
        are popped from the map.

    :param remove_initializers: Whether to pop the consumed initializers.

    :return: Tuple ``(alpha, beta, trans_a, trans_b, weight, bias)``.
    """
    attrs = get_onnx_attrs(node, initializers)
    alpha = attrs["alpha"]
    beta = attrs["beta"]
    trans_a = attrs["transA"]
    trans_b = attrs["transB"]
    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]

    return alpha, beta, trans_a, trans_b, weight, bias


def _get_conv_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract Conv parameters and the resolved attribute map.

    Supports both regular convolutions (group=1) and grouped / depthwise
    convolutions (group > 1). Missing bias is materialised as a zero
    vector matched to ``weight.shape[0]`` so callers always see a real
    array.

    :param node: Conv NodeProto.

    :param initializers: Initializer map keyed by tensor name. When
        ``remove_initializers`` is true the weight (and bias when present)
        initializers are popped from the map.

    :param remove_initializers: Whether to pop the consumed initializers.

    :return: Tuple ``(weight, bias, attrs)``.
    """
    attrs = get_onnx_attrs(node, initializers)
    kernel_size = attrs["kernel_shape"]

    if len(kernel_size) != 2:
        raise NotImplementedError(f"Unsupported kernel_size={kernel_size} for Conv/ConvTranspose.")

    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    if remove_initializers:
        del initializers[node.input[1]]

    if len(node.input) == 2:  # No bias
        bias = np.zeros(weight.shape[0], dtype=weight.dtype)
    else:
        bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
        if remove_initializers:
            del initializers[node.input[2]]

    return weight, bias, attrs


def _get_conv_transpose_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract ConvTranspose parameters and the resolved attribute map.

    Restricted to ``group == 1`` and 2-D kernels; grouped ConvTranspose
    fusion is not supported yet. Missing bias is materialised as a zero
    vector matched to ``weight.shape[1]`` so callers always see a real
    array.

    :param node: ConvTranspose NodeProto.

    :param initializers: Initializer map keyed by tensor name. When
        ``remove_initializers`` is true the weight (and bias when present)
        initializers are popped from the map.

    :param remove_initializers: Whether to pop the consumed initializers.

    :return: Tuple ``(weight, bias, attrs)``.
    """
    attrs = get_onnx_attrs(node, initializers)
    kernel_size = attrs["kernel_shape"]
    group = attrs["group"]

    if len(kernel_size) != 2:
        raise NotImplementedError(f"Unsupported kernel_size={kernel_size} for Conv/ConvTranspose.")
    if group != 1:
        raise NotImplementedError(f"Unsupported group={group} for ConvTranspose.")

    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    if remove_initializers:
        del initializers[node.input[1]]

    if len(node.input) == 2:  # No bias
        bias = np.zeros(weight.shape[1], dtype=weight.dtype)
    else:
        bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
        if remove_initializers:
            del initializers[node.input[2]]

    return weight, bias, attrs


def compute_batchnorm_fusion_params(
    epsilon: float,
    scale: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    target_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the fused BatchNorm scale and bias for a downstream linear op.

    This shared computation is used in Conv-BN, Gemm-BN, and Transpose-BN
    fusions.

    :param epsilon: BatchNorm epsilon value.

    :param scale: BatchNorm scale parameter.

    :param bias: BatchNorm bias parameter.

    :param mean: BatchNorm mean parameter.

    :param var: BatchNorm variance parameter.

    :param target_dtype: Target dtype to preserve precision (default: float32).

    :return: Tuple of (bn_weight, bn_bias) for fusion.
    """
    if target_dtype is None:
        target_dtype = np.dtype(np.float32)

    # Cast all inputs to target dtype to avoid float32 / float64 mismatch.
    scale = scale.astype(target_dtype, copy=False)
    bias = bias.astype(target_dtype, copy=False)
    mean = mean.astype(target_dtype, copy=False)
    var = var.astype(target_dtype, copy=False)

    bn_weight = scale / np.sqrt(var + target_dtype.type(epsilon))
    bn_bias = bias - mean * bn_weight
    return bn_weight, bn_bias
