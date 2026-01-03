__docformat__ = "restructuredtext"
__all__ = [
    "_get_batchnorm_params",
    "_get_conv_params",
    "_get_conv_transpose_params",
    "_get_gemm_params",
    "_is_only_next_node",
    "compute_batchnorm_fusion_params",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._onnx_attrs import get_onnx_attrs


def _is_only_next_node(pre_node: NodeProto, cur_node: NodeProto, nodes: list[NodeProto]) -> bool:
    """Check the pre_node and node are in a single path.

    Because if there are multiple paths, we cannot fuse the nodes to avoid changing
    the computation graph.
    """
    pre_node_name = pre_node.output[0]
    return all(not (pre_node_name in node.input and node != cur_node) for node in nodes)


def _get_batchnorm_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the parameters of a BatchNormalization node."""
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
    """Get the parameters of a Gemm node."""
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
):
    """Get the parameters of a Conv or ConvTranspose node.

    Supports both regular convolutions (group=1) and grouped/depthwise
    convolutions (group > 1).
    """
    attrs = get_onnx_attrs(node, initializers)
    kernel_size = attrs["kernel_shape"]
    attrs["group"]

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
):
    """Get the parameters of a ConvTranspose node."""
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
    """Compute fused BatchNorm weight and bias for fusion operations.

    This shared computation is used in Conv-BN, Gemm-BN, and Transpose-BN fusions.

    :param epsilon: BatchNorm epsilon value
    :param scale: BatchNorm scale parameter
    :param bias: BatchNorm bias parameter
    :param mean: BatchNorm mean parameter
    :param var: BatchNorm variance parameter
    :param target_dtype: Target dtype to preserve precision (default: float32)
    :return: Tuple of (bn_weight, bn_bias) for fusion
    """
    # Use default dtype if not specified
    if target_dtype is None:
        target_dtype = np.dtype(np.float32)

    # Cast all inputs to target dtype to avoid float32/float64 mismatch
    scale = scale.astype(target_dtype, copy=False)
    bias = bias.astype(target_dtype, copy=False)
    mean = mean.astype(target_dtype, copy=False)
    var = var.astype(target_dtype, copy=False)

    # Perform computation in target dtype
    bn_weight = scale / np.sqrt(var + target_dtype.type(epsilon))
    bn_bias = bias - mean * bn_weight
    return bn_weight, bn_bias
