__docformat__ = ["restructuredtext"]
__all__ = [
    "VERBOSE",
    "_is_only_next_node",
    "_get_batchnorm_params",
    "_get_gemm_params",
    "_get_conv_params",
    "_get_convtranspose_params",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from ..onnx_attrs import get_onnx_attrs

VERBOSE = False


def _is_only_next_node(
    pre_node: NodeProto, cur_node: NodeProto, nodes: list[NodeProto]
) -> bool:
    """
    Check the pre_node and node are in a single path. Because if there are multiple
    paths, we cannot fuse the nodes to avoid changing the computation graph.
    """
    pre_node_name = pre_node.output[0]
    for node in nodes:
        if pre_node_name in node.input and node != cur_node:
            return False
    return True


def _get_batchnorm_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the parameters of a BatchNormalization node.
    """
    attrs = get_onnx_attrs(node, initializers)
    epsilon = attrs["epsilon"]
    scale = onnx.numpy_helper.to_array(initializers[node.input[1]])
    b = onnx.numpy_helper.to_array(initializers[node.input[2]])
    mean = onnx.numpy_helper.to_array(initializers[node.input[3]])
    var = onnx.numpy_helper.to_array(initializers[node.input[4]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]
        del initializers[node.input[3]]
        del initializers[node.input[4]]

    return epsilon, scale, b, mean, var


def _get_gemm_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    """
    Get the parameters of a Gemm node.
    """
    attrs = get_onnx_attrs(node, initializers)
    alpha = attrs["alpha"]
    beta = attrs["beta"]
    transA = attrs["transA"]
    transB = attrs["transB"]
    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
    if remove_initializers:
        del initializers[node.input[1]]
        del initializers[node.input[2]]

    return alpha, beta, transA, transB, weight, bias


def _get_conv_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
):
    """
    Get the parameters of a Conv or ConvTranspose node.
    """
    attrs = get_onnx_attrs(node, initializers)
    kernel_size = attrs["kernel_shape"]
    group = attrs["group"]

    if len(kernel_size) != 2:
        raise NotImplementedError(
            f"Unsupported kernel_size={kernel_size} for Conv/ConvTranspose."
        )
    if group != 1:
        raise NotImplementedError(f"Unsupported group={group} for Conv.")

    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    if remove_initializers:
        del initializers[node.input[1]]

    if len(node.input) == 2:  # No bias
        bias = np.zeros(weight.shape[0])
    else:
        bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
        if remove_initializers:
            del initializers[node.input[2]]

    return weight, bias, attrs


def _get_convtranspose_params(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    remove_initializers: bool = False,
):
    """
    Get the parameters of a ConvTranspose node.
    """
    attrs = get_onnx_attrs(node, initializers)
    kernel_size = attrs["kernel_shape"]
    group = attrs["group"]

    if len(kernel_size) != 2:
        raise NotImplementedError(
            f"Unsupported kernel_size={kernel_size} for Conv/ConvTranspose."
        )
    if group != 1:
        raise NotImplementedError(f"Unsupported group={group} for ConvTranspose.")

    weight = onnx.numpy_helper.to_array(initializers[node.input[1]])
    if remove_initializers:
        del initializers[node.input[1]]

    if len(node.input) == 2:  # No bias
        bias = np.zeros(weight.shape[1])
    else:
        bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
        if remove_initializers:
            del initializers[node.input[2]]

    return weight, bias, attrs
