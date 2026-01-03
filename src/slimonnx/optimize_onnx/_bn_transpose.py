__docformat__ = "restructuredtext"
__all__ = ["_fuse_transpose_batchnorm_transpose"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._constants import TRANSPOSE_CHW_TO_CWH
from slimonnx.optimize_onnx._onnx_attrs import get_onnx_attrs
from slimonnx.optimize_onnx._utils import (
    _get_batchnorm_params,
    _is_only_next_node,
    compute_batchnorm_fusion_params,
)


def _validate_transpose_perm(tp_node: NodeProto, initializers: dict[str, TensorProto]) -> None:
    """Validate that transpose node has expected permutation.

    :param tp_node: Transpose node
    :param initializers: Dictionary of initializers
    :raises ValueError: If permutation is invalid
    """
    perm = get_onnx_attrs(tp_node, initializers)["perm"]
    if not all(p_i == p_j for p_i, p_j in zip(perm, TRANSPOSE_CHW_TO_CWH, strict=False)):
        raise ValueError(
            f"Transpose node {tp_node.name} has unsupported perm={perm}. "
            f"Expected {TRANSPOSE_CHW_TO_CWH}."
        )


def _can_fuse_to_gemm(input_name: str, input_nodes: list | None, data_shapes: dict | None) -> bool:
    """Check if input tensor has rank 2 (required for Gemm fusion).

    :param input_name: Input tensor name
    :param input_nodes: List of graph input nodes
    :param data_shapes: Dictionary of tensor shapes
    :return: True if can fuse to Gemm (rank 2), False otherwise
    """
    if data_shapes is not None and input_name in data_shapes:
        input_shape = data_shapes[input_name]
        return len(input_shape) == 2

    if input_nodes is not None:
        for graph_input in input_nodes:
            if graph_input.name == input_name:
                input_rank = len(graph_input.type.tensor_type.shape.dim)
                return input_rank == 2

    return True


def _create_gemm_fusion(
    bn_node: NodeProto,
    tp_node1: NodeProto,
    tp_node2: NodeProto,
    weight: np.ndarray,
    bias: np.ndarray,
    initializers: dict[str, TensorProto],
) -> NodeProto:
    """Create Gemm node for rank-2 fusion.

    :param bn_node: BatchNormalization node
    :param tp_node1: First Transpose node
    :param tp_node2: Second Transpose node
    :param weight: Fused weight matrix
    :param bias: Fused bias vector
    :param initializers: Dictionary of initializers
    :return: Gemm node
    """
    weight_name = bn_node.input[1] + "_gemm"
    bias_name = bn_node.input[2] + "_gemm"

    weight_tensor = onnx.numpy_helper.from_array(weight, weight_name)
    bias_tensor = onnx.numpy_helper.from_array(bias, bias_name)

    initializers[weight_name] = weight_tensor
    initializers[bias_name] = bias_tensor

    return onnx.NodeProto(
        op_type="Gemm",
        input=[tp_node1.input[0], weight_name, bias_name],
        output=tp_node2.output,
        name=bn_node.name + "_gemm",
    )


def _create_matmul_add_fusion(
    bn_node: NodeProto,
    tp_node1: NodeProto,
    tp_node2: NodeProto,
    weight: np.ndarray,
    bias: np.ndarray,
    initializers: dict[str, TensorProto],
) -> tuple[NodeProto, NodeProto]:
    """Create MatMul+Add nodes for non-rank-2 fusion.

    :param bn_node: BatchNormalization node
    :param tp_node1: First Transpose node
    :param tp_node2: Second Transpose node
    :param weight: Fused weight matrix
    :param bias: Fused bias vector
    :param initializers: Dictionary of initializers
    :return: Tuple of (MatMul node, Add node)
    """
    weight_name = bn_node.input[1] + "_matmul"
    bias_name = bn_node.input[2] + "_add"
    matmul_output = bn_node.name + "_matmul_output"

    weight_tensor = onnx.numpy_helper.from_array(weight, weight_name)
    bias_tensor = onnx.numpy_helper.from_array(bias, bias_name)

    initializers[weight_name] = weight_tensor
    initializers[bias_name] = bias_tensor

    matmul_node = onnx.NodeProto(
        op_type="MatMul",
        input=[tp_node1.input[0], weight_name],
        output=[matmul_output],
        name=bn_node.name + "_matmul",
    )

    add_node = onnx.NodeProto(
        op_type="Add",
        input=[matmul_output, bias_name],
        output=tp_node2.output,
        name=bn_node.name + "_add",
    )

    return matmul_node, add_node


def _fuse_transpose_batchnorm_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list | None = None,
    data_shapes: dict | None = None,
) -> list[NodeProto]:
    """Fuse Transpose-BatchNormalization-Transpose pattern to Gemm or MatMul+Add.

    For rank-2 inputs: converts to Gemm
    For non-rank-2 inputs: converts to MatMul+Add

    :param nodes: List of nodes in the graph
    :param initializers: Dictionary of initializers
    :param input_nodes: List of graph input nodes (optional)
    :param data_shapes: Dictionary of tensor shapes (optional)
    :return: Optimized list of nodes
    """
    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        new_node = node
        if (
            node.op_type == "Transpose"
            and pre_pre_node is not None
            and pre_node is not None
            and pre_pre_node.op_type == "Transpose"
            and pre_node.op_type == "BatchNormalization"
            and _is_only_next_node(pre_pre_node, pre_node, nodes)
            and _is_only_next_node(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()
            tp_node1, bn_node, tp_node2 = pre_pre_node, pre_node, node

            _validate_transpose_perm(tp_node1, initializers)
            _validate_transpose_perm(tp_node2, initializers)

            epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
                bn_node, initializers, remove_initializers=True
            )

            input_name = tp_node1.input[0]
            can_fuse = _can_fuse_to_gemm(input_name, input_nodes, data_shapes)

            # Preserve dtype from scale tensor to avoid float32/float64 mismatch
            target_dtype = scale.dtype
            bn_weight, bias = compute_batchnorm_fusion_params(
                epsilon, scale, bn_param_bias, mean, var, target_dtype
            )
            weight = np.diag(bn_weight).astype(target_dtype, copy=False)

            if can_fuse:
                new_node = _create_gemm_fusion(
                    bn_node, tp_node1, tp_node2, weight, bias, initializers
                )
            else:
                matmul_node, add_node = _create_matmul_add_fusion(
                    bn_node, tp_node1, tp_node2, weight, bias, initializers
                )
                new_nodes.append(matmul_node)
                new_node = add_node

        new_nodes.append(new_node)
        pre_pre_node = pre_node
        pre_node = node

    return new_nodes
