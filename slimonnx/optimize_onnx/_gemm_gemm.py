__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_gemm_gemm"]

import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ._utils import *


def _fuse_gemm_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    all_gemm_nodes = [node for node in nodes if node.op_type == "Gemm"]
    all_gemm_output_names = [node.output[0] for node in all_gemm_nodes]
    # Reverse the order and group the gemm nodes by their pre nodes
    all_gemm_nodes = all_gemm_nodes[::-1]

    # Find all adjacent gemm nodes group
    fused_gemm_groups = []
    i = 0
    group = []
    while True:
        if i >= len(all_gemm_nodes):
            break
        node = all_gemm_nodes[i]
        i += 1
        assert node.input[1] in initializers

        group.append(node)
        if node.input[0] in all_gemm_output_names:
            continue

        fused_gemm_groups.append(group[::-1])
        group = []

    fused_gemm_groups = fused_gemm_groups[::-1]

    # for group in fused_gemm_groups:
    #     print([e.output[0] for e in group])

    reserved_gemm = {group[0].output[0]: i for i, group in enumerate(fused_gemm_groups)}

    new_nodes = []
    for node in nodes:
        if node.op_type == "Gemm":
            if not node.output[0] in reserved_gemm:
                continue

            group = fused_gemm_groups[reserved_gemm[node.output[0]]]

            # Fuse all the gemm nodes in the group
            # Collect all weight and bias initializers in the group
            all_weights = []
            all_biases = []
            for gemm_node in group:
                assert gemm_node.input[1] in initializers
                assert gemm_node.input[2] in initializers
                alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                    gemm_node, initializers, True
                )
                assert alpha == beta == 1
                assert transA == transB == 0
                all_weights.append(weight)
                all_biases.append(bias)

            # for w, b in zip(all_weights, all_biases):
            #     print(w.shape, b.shape)

            # Create the new weight and bias initializers
            new_weight = all_weights[0]
            new_bias = all_biases[0]
            for i in range(1, len(all_weights)):
                new_weight = new_weight @ all_weights[i]
                new_bias = new_bias @ all_weights[i] + all_biases[i]

            new_weight = onnx.numpy_helper.from_array(new_weight, group[0].input[1])
            new_bias = onnx.numpy_helper.from_array(new_bias, group[0].input[2])
            initializers[group[0].input[1]] = new_weight
            initializers[group[0].input[2]] = new_bias

            new_gemm_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=[group[0].input[0], group[0].input[1], group[0].input[2]],
                outputs=group[-1].output,
                name=group[0].name,
            )
            new_nodes.append(new_gemm_node)
            continue

        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Fused {len(fused_gemm_groups)} Gemm-Gemm groups.")

    return new_nodes
