__docformat__ = "restructuredtext"
__all__ = ["_fuse_gemm_gemm"]

import onnx
from onnx import NodeProto, TensorProto


from ._utils import _get_gemm_params


def _fuse_gemm_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    # Get all gemm nodes
    # Remove those gemm nodes that has multiple pre/post nodes
    chosen_gemm_nodes = [node for node in nodes if node.op_type == "Gemm"]
    new_all_gemm_nodes = []
    for gemm_node in chosen_gemm_nodes:
        n_gemm_pre_nodes = 0
        n_gemm_post_nodes = 0
        for node in nodes:
            for output_i in node.output:
                if output_i in gemm_node.input:
                    n_gemm_pre_nodes += 1
            if gemm_node.output[0] in node.input:
                n_gemm_post_nodes += 1
        if not (n_gemm_pre_nodes > 1 or n_gemm_post_nodes > 1):
            # If the pre or post nodes are 0, that means the gemm node is connected to
            # the input or output of the model, so we can keep it
            new_all_gemm_nodes.append(gemm_node)
    chosen_gemm_nodes = new_all_gemm_nodes
    # Reverse the order and group the gemm nodes by their pre nodes
    chosen_gemm_nodes = chosen_gemm_nodes[::-1]

    chosen_gemm_output_names = [node.output[0] for node in chosen_gemm_nodes]
    # Find all adjacent gemm nodes group
    fused_gemm_groups = []
    i = 0
    group = []
    while True:
        if i >= len(chosen_gemm_nodes):
            break

        node = chosen_gemm_nodes[i]
        i += 1
        if node.input[1] not in initializers:
            raise ValueError(
                f"Gemm node {node.name} weight input {node.input[1]} not found in initializers."
            )

        group.append(node)
        if node.input[0] in chosen_gemm_output_names:
            continue

        if len(group) > 1:
            fused_gemm_groups.append(group[::-1])
        group = []

    fused_gemm_groups = fused_gemm_groups[::-1]
    fused_gemm_nodes_output_names = [
        node.output[0] for group in fused_gemm_groups for node in group
    ]
    # for group in fused_gemm_groups:
    #     print([e.output[0] for e in group])

    gemm_group_start_nodes = {
        group[0].output[0]: i for i, group in enumerate(fused_gemm_groups)
    }

    new_nodes = []
    for node in nodes:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue
        if node.output[0] not in fused_gemm_nodes_output_names:
            new_nodes.append(node)
            continue
        if node.output[0] not in gemm_group_start_nodes:
            continue

        group = fused_gemm_groups[gemm_group_start_nodes[node.output[0]]]

        # Fuse all the gemm nodes in the group
        # Collect all weight and bias initializers in the group
        all_weights = []
        all_biases = []
        for gemm_node in group:
            if gemm_node.input[1] not in initializers:
                raise ValueError(
                    f"Gemm node {gemm_node.name} weight input {gemm_node.input[1]} not found in initializers."
                )
            if gemm_node.input[2] not in initializers:
                raise ValueError(
                    f"Gemm node {gemm_node.name} bias input {gemm_node.input[2]} not found in initializers."
                )
            alpha, beta, transA, transB, weight, bias = _get_gemm_params(
                gemm_node, initializers, True
            )
            if alpha != 1 or beta != 1:
                raise ValueError(
                    f"Gemm node {gemm_node.name} has unsupported alpha={alpha} or beta={beta}. Only alpha=beta=1 is supported for fusion."
                )
            if transA != 0 or transB != 0:
                raise ValueError(
                    f"Gemm node {gemm_node.name} has unsupported transA={transA} or transB={transB}. Only transA=transB=0 is supported for fusion."
                )
            all_weights.append(weight)
            all_biases.append(bias)

        # for w, b in zip(all_weights, all_biases):
        #     print(w.shape, b.shape)

        # Merge all weights and biases
        new_weight = all_weights[0]
        new_bias = all_biases[0]
        for i in range(1, len(all_weights)):
            new_weight = new_weight @ all_weights[i]
            new_bias = new_bias @ all_weights[i] + all_biases[i]

        # Create new initializers for the merged weights and biases
        new_weight_name = group[0].input[1]
        new_bias_name = group[0].input[2]
        new_weight = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        new_bias = onnx.numpy_helper.from_array(new_bias, new_bias_name)
        initializers[new_weight_name] = new_weight
        initializers[new_bias_name] = new_bias

        # Create a new gemm node with the merged weights and biases
        new_gemm_node = onnx.helper.make_node(
            op_type="Gemm",
            inputs=[group[0].input[0], new_weight_name, new_bias_name],
            outputs=group[-1].output,
            name=group[-1].name,
        )

        new_nodes.append(new_gemm_node)

    return new_nodes
