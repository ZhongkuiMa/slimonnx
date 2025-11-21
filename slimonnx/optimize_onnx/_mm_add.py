__docformat__ = "restructuredtext"
__all__ = ["_fuse_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto

from ._utils import _is_only_next_node


def _fuse_matmul_add(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list = None,
    data_shapes: dict = None,
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

            # Check if input tensor has rank != 2 (Gemm requires exactly rank 2)
            if can_fuse and data_shapes is not None and input_name in data_shapes:
                input_shape = data_shapes[input_name]
                if len(input_shape) != 2:
                    can_fuse = False
            # Fallback: check graph inputs if shapes not available
            elif can_fuse and input_nodes is not None:
                for graph_input in input_nodes:
                    if graph_input.name == input_name:
                        input_rank = len(graph_input.type.tensor_type.shape.dim)
                        if input_rank != 2:
                            can_fuse = False
                        break

            if can_fuse:
                new_nodes.pop()
                inputs = (
                    [input_name, weight_name, bias_name]
                    if transB == 0
                    else [weight_name, input_name, bias_name]
                )
                new_node = onnx.NodeProto(
                    op_type="Gemm", input=inputs, output=add_node.output
                )

        new_nodes.append(new_node)
        pre_node = node

    return new_nodes
