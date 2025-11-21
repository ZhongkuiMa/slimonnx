__docformat__ = "restructuredtext"
__all__ = ["_fuse_transpose_batchnorm_transpose"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from ._utils import (
    _is_only_next_node,
    _get_batchnorm_params,
    compute_batchnorm_fusion_params,
)
from ..onnx_attrs import get_onnx_attrs
from .constants import TRANSPOSE_CHW_TO_CWH, GEMM_REQUIRED_RANK


def _fuse_transpose_batchnorm_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list = None,
    data_shapes: dict = None,
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

            perm1 = get_onnx_attrs(tp_node1, initializers)["perm"]
            perm2 = get_onnx_attrs(tp_node2, initializers)["perm"]
            if not all(p_i == p_j for p_i, p_j in zip(perm1, TRANSPOSE_CHW_TO_CWH)):
                raise ValueError(
                    f"First Transpose node {tp_node1.name} has unsupported perm={perm1}. Expected {TRANSPOSE_CHW_TO_CWH}."
                )
            if not all(p_i == p_j for p_i, p_j in zip(perm2, TRANSPOSE_CHW_TO_CWH)):
                raise ValueError(
                    f"Second Transpose node {tp_node2.name} has unsupported perm={perm2}. Expected {TRANSPOSE_CHW_TO_CWH}."
                )

            epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
                bn_node, initializers, True
            )

            # Check if input tensor has rank != 2 (Gemm requires exactly rank 2)
            input_name = tp_node1.input[0]
            can_fuse = True

            if data_shapes is not None and input_name in data_shapes:
                input_shape = data_shapes[input_name]
                if len(input_shape) != 2:
                    can_fuse = False
            # Fallback: check graph inputs if shapes not available
            elif input_nodes is not None:
                for graph_input in input_nodes:
                    if graph_input.name == input_name:
                        input_rank = len(graph_input.type.tensor_type.shape.dim)
                        if input_rank != 2:
                            can_fuse = False
                        break

            # Preserve dtype from scale tensor to avoid float32/float64 mismatch
            target_dtype = scale.dtype
            bn_weight, bias = compute_batchnorm_fusion_params(
                epsilon, scale, bn_param_bias, mean, var, target_dtype
            )
            weight = np.diag(bn_weight).astype(target_dtype, copy=False)

            if can_fuse:
                # Rank-2 input: convert to Gemm
                weight_name = bn_node.input[1] + "_gemm"
                bias_name = bn_node.input[2] + "_gemm"

                weight_tensor = onnx.numpy_helper.from_array(weight, weight_name)
                bias_tensor = onnx.numpy_helper.from_array(bias, bias_name)

                initializers[weight_name] = weight_tensor
                initializers[bias_name] = bias_tensor

                new_node = onnx.NodeProto(
                    op_type="Gemm",
                    input=[tp_node1.input[0], weight_name, bias_name],
                    output=tp_node2.output,
                    name=bn_node.name + "_gemm",
                )

            else:
                # Non-rank-2 input: convert to MatMul+Add
                weight_name = bn_node.input[1] + "_matmul"
                bias_name = bn_node.input[2] + "_add"
                matmul_output = bn_node.name + "_matmul_output"

                weight_tensor = onnx.numpy_helper.from_array(weight, weight_name)
                bias_tensor = onnx.numpy_helper.from_array(bias, bias_name)

                initializers[weight_name] = weight_tensor
                initializers[bias_name] = bias_tensor

                # Create MatMul node
                matmul_node = onnx.NodeProto(
                    op_type="MatMul",
                    input=[tp_node1.input[0], weight_name],
                    output=[matmul_output],
                    name=bn_node.name + "_matmul",
                )
                new_nodes.append(matmul_node)

                # Create Add node
                new_node = onnx.NodeProto(
                    op_type="Add",
                    input=[matmul_output, bias_name],
                    output=tp_node2.output,
                    name=bn_node.name + "_add",
                )

        new_nodes.append(new_node)
        pre_pre_node = pre_node
        pre_node = node

    return new_nodes
