__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_gemm_gemm"]

from typing import Any

import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ._utils import *


def _fuse_gemm_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    new_nodes = []

    all_gemm_nodes = [node.output[0] for node in nodes if node.op_type == "Gemm"]
    is_next_node_gemm: dict[str, Any] = {node_name: [] for node_name in all_gemm_nodes}
    for node in nodes:
        is_gemm = node.op_type == "Gemm"
        for input_i in node.input:
            if input_i in all_gemm_nodes:
                is_next_node_gemm[input_i].append(is_gemm)
    for node_name, is_next_node_gemm_i in is_next_node_gemm.items():
        is_next_node_gemm[node_name] = (
            all(is_next_node_gemm_i) and len(is_next_node_gemm_i) > 0
        )

    gemm_node_names_to_fuse = [
        node_name
        for node_name, is_next_node_gemm in is_next_node_gemm.items()
        if is_next_node_gemm
    ]
    gemm_nodes_to_fuse = {
        node.output[0]: node
        for node in nodes
        if node.output[0] in gemm_node_names_to_fuse
    }

    count = 0

    for node in nodes:
        new_node = node
        if node.op_type == "Gemm":
            if node.output[0] in gemm_nodes_to_fuse:
                continue
            elif node.input[0] in gemm_nodes_to_fuse:
                # Fuse the current node with the previous node
                pre_node = gemm_nodes_to_fuse[node.input[0]]
                data_type = initializers[node.input[1]].data_type
                alpha1, beta1, transA1, transB1, weight1, bias1 = _get_gemm_params(
                    node, initializers
                )
                alpha2, beta2, transA2, transB2, weight2, bias2 = _get_gemm_params(
                    pre_node, initializers, remove_initializers=False
                )
                assert alpha1 == alpha2 == beta1 == beta2 == 1
                assert transA1 == transA2 == transB1 == transB2 == 0

                """
                IDEA
                   (X @ W_2 + b_2) @ W_1 + b_1
                => X @ (W_2 @ W_1) + (b_2 @ W_1 + b_1)
                """
                new_weight = weight2 @ weight1
                new_bias = bias2 @ weight1 + bias1

                new_weight = onnx.helper.make_tensor(
                    name=node.input[1],
                    data_type=data_type,
                    dims=new_weight.shape,
                    vals=new_weight.flatten().tolist(),
                )
                new_bias = onnx.helper.make_tensor(
                    name=node.input[2],
                    data_type=data_type,
                    dims=new_bias.shape,
                    vals=new_bias.flatten().tolist(),
                )
                initializers[node.input[1]] = new_weight
                initializers[node.input[2]] = new_bias

                new_node = onnx.helper.make_node(
                    op_type="Gemm",
                    inputs=[pre_node.input[0], node.input[1], node.input[2]],
                    outputs=node.output,
                    name=node.name,
                )

                count += 1

        new_nodes.append(new_node)

    # Remove the initializers of the fused nodes
    for node in gemm_nodes_to_fuse.values():
        del initializers[node.input[1]]
        del initializers[node.input[2]]

    if utils.VERBOSE:
        print(f"Fused {count} Gemm-Gemm nodes.")

    return new_nodes
