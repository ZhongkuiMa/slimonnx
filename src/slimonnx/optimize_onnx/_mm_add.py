__docformat__ = "restructuredtext"
__all__ = ["_fuse_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import _is_only_next_node


def _extract_matmul_add_params(
    matmul_node: NodeProto, add_node: NodeProto, initializers: dict[str, TensorProto]
) -> tuple[str, str, str, int]:
    """Extract parameters from MatMul and Add nodes.

    :param matmul_node: MatMul node
    :param add_node: Add node
    :param initializers: Dictionary of initializers
    :return: Tuple of (input_name, weight_name, bias_name, trans_b)
    """
    input_name, weight_name, trans_b = (
        (matmul_node.input[0], matmul_node.input[1], 0)
        if matmul_node.input[1] in initializers
        else (matmul_node.input[1], matmul_node.input[0], 1)
    )
    bias_name = (
        add_node.input[1] if add_node.input[0] == matmul_node.output[0] else add_node.input[0]
    )
    return input_name, weight_name, bias_name, trans_b


def _can_fuse_to_gemm_matmul_add(
    input_name: str,
    weight_name: str,
    bias_name: str,
    initializers: dict[str, TensorProto],
    input_nodes: list | None,
    data_shapes: dict | None,
) -> bool:
    """Check if MatMul+Add can be fused to Gemm.

    :param input_name: Input tensor name
    :param weight_name: Weight tensor name
    :param bias_name: Bias tensor name
    :param initializers: Dictionary of initializers
    :param input_nodes: List of graph input nodes
    :param data_shapes: Dictionary of tensor shapes
    :return: True if can fuse to Gemm
    """
    weight_dim = len(initializers[weight_name].dims)
    if weight_dim != 2:
        return False

    bias_dim = len(initializers[bias_name].dims)
    if bias_dim != 1:
        return False

    # Check if input tensor has rank 2 (required for Gemm)
    if data_shapes is not None and input_name in data_shapes:
        input_shape = data_shapes[input_name]
        return len(input_shape) == 2

    if input_nodes is not None:
        for graph_input in input_nodes:
            if graph_input.name == input_name:
                input_rank = len(graph_input.type.tensor_type.shape.dim)
                return input_rank == 2

    return True


def _fuse_matmul_add(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list | None = None,
    data_shapes: dict | None = None,
) -> list[NodeProto]:
    """Fuse a MatMul and an Add node into a single Gemm node.

    Note: Gemm requires exactly rank 2 inputs, so we check tensor shapes to avoid
    fusing MatMul with non-rank-2 inputs.

    :param nodes: List of nodes in the graph
    :param initializers: Dictionary of initializers
    :param input_nodes: List of graph input nodes (optional)
    :param data_shapes: Dictionary of tensor shapes (optional)
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
            and (pre_node.input[0] in initializers or pre_node.input[1] in initializers)
            and _is_only_next_node(pre_node, node, nodes)
        ):
            matmul_node, add_node = pre_node, node
            input_name, weight_name, bias_name, trans_b = _extract_matmul_add_params(
                matmul_node, add_node, initializers
            )

            can_fuse = _can_fuse_to_gemm_matmul_add(
                input_name, weight_name, bias_name, initializers, input_nodes, data_shapes
            )

            if can_fuse:
                new_nodes.pop()
                inputs = (
                    [input_name, weight_name, bias_name]
                    if trans_b == 0
                    else [weight_name, input_name, bias_name]
                )
                new_node = onnx.NodeProto(op_type="Gemm", input=inputs, output=add_node.output)

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
