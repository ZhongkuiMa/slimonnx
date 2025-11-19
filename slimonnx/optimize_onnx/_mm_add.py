__docformat__ = "restructuredtext"
__all__ = ["_fuse_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto

from ._utils import _is_only_next_node


def _fuse_matmul_add(
    nodes: list[NodeProto], initializers: dict[str, TensorProto], verbose: bool = False
) -> list[NodeProto]:
    """
    Fuse a MatMul and an Add node into a single Gemm node.
    """
    count = 0

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
            count += 1

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

            if can_fuse:
                new_nodes.pop()
                inputs = (
                    [input_name, weight_name, bias_name]
                    if transB == 0
                    else [weight_name, input_name, bias_name]
                )
                new_node = onnx.helper.make_node(
                    op_type="Gemm", inputs=inputs, outputs=add_node.output
                )

        new_nodes.append(new_node)
        pre_node = node

    if verbose:
        print(f"Fused {count} MatMul-Add nodes.")

    return new_nodes
