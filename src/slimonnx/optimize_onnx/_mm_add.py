"""Fuse MatMul+Add sequences into Gemm operators."""

__docformat__ = "restructuredtext"
__all__ = ["_fuse_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import _is_only_next_node


def _extract_matmul_add_params(
    matmul_node: NodeProto, add_node: NodeProto, initializers: dict[str, TensorProto]
) -> tuple[str, str, str] | None:
    """Extract parameters from MatMul and Add nodes.

    :param matmul_node: MatMul node.

    :param add_node: Add node.

    :param initializers: Dictionary of initializers.

    :return: ``(input_name, weight_name, bias_name)`` when the right MatMul
        operand is static, otherwise ``None``. A constant-left MatMul is not a
        linear-layer Gemm pattern and must keep its operand order.
    """
    if matmul_node.input[1] not in initializers:
        return None
    input_name, weight_name = matmul_node.input
    bias_name = (
        add_node.input[1] if add_node.input[0] == matmul_node.output[0] else add_node.input[0]
    )
    return input_name, weight_name, bias_name


def _can_fuse_to_gemm_matmul_add(
    input_name: str,
    weight_name: str,
    bias_name: str,
    initializers: dict[str, TensorProto],
    input_nodes: list | None,
    data_shapes: dict | None,
) -> bool:
    """Check if MatMul+Add can be fused to Gemm.

    :param input_name: Input tensor name.

    :param weight_name: Weight tensor name.

    :param bias_name: Bias tensor name.

    :param initializers: Dictionary of initializers.

    :param input_nodes: List of graph input nodes.

    :param data_shapes: Dictionary of tensor shapes.

    :return: True if can fuse to Gemm
    """
    weight_shape = tuple(initializers[weight_name].dims)
    bias_shape = tuple(initializers[bias_name].dims)
    if len(weight_shape) != 2 or len(bias_shape) != 1:
        return False

    input_shape = None
    if data_shapes is not None and input_name in data_shapes:
        input_shape = tuple(data_shapes[input_name])
    elif input_nodes is not None:
        for graph_input in input_nodes:
            if graph_input.name == input_name:
                dims = graph_input.type.tensor_type.shape.dim
                if any(not dim.HasField("dim_value") for dim in dims):
                    return False
                input_shape = tuple(dim.dim_value for dim in dims)
                break

    if input_shape is None or len(input_shape) != 2:
        return False
    return bool(input_shape[1] == weight_shape[0] and bias_shape[0] == weight_shape[1])


def _fuse_matmul_add(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list | None = None,
    data_shapes: dict | None = None,
) -> list[NodeProto]:
    """Fuse a MatMul and an Add node into a single Gemm node.

    Note: Gemm requires exactly rank 2 inputs, so we check tensor shapes to avoid
    fusing MatMul with non-rank-2 inputs.

    :param nodes: List of nodes in the graph.

    :param initializers: Dictionary of initializers.

    :param input_nodes: List of graph input nodes (optional).

    :param data_shapes: Dictionary of tensor shapes (optional).

    :return: Optimized list of nodes
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
            and pre_node.input[1] in initializers
            and _is_only_next_node(pre_node, node, nodes)
        ):
            matmul_node, add_node = pre_node, node
            params = _extract_matmul_add_params(matmul_node, add_node, initializers)
            assert params is not None
            input_name, weight_name, bias_name = params

            can_fuse = _can_fuse_to_gemm_matmul_add(
                input_name, weight_name, bias_name, initializers, input_nodes, data_shapes
            )

            if can_fuse:
                new_nodes.pop()
                new_node = onnx.NodeProto()
                new_node.CopyFrom(add_node)
                new_node.op_type = "Gemm"
                del new_node.input[:]
                new_node.input.extend([input_name, weight_name, bias_name])
                del new_node.attribute[:]

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
