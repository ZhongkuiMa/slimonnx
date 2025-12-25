__docformat__ = "restructuredtext"
__all__ = ["_fuse_gemm_gemm"]

import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import _get_gemm_params


def _count_node_connections(gemm_node: NodeProto, nodes: list[NodeProto]) -> tuple[int, int]:
    """Count predecessor and successor connections for a Gemm node.

    :param gemm_node: Gemm node to check
    :param nodes: All nodes in the graph
    :return: Tuple of (n_predecessors, n_successors)
    """
    n_pre = 0
    n_post = 0
    for node in nodes:
        for output_i in node.output:
            if output_i in gemm_node.input:
                n_pre += 1
        if gemm_node.output[0] in node.input:
            n_post += 1
    return n_pre, n_post


def _filter_fusable_gemm_nodes(nodes: list[NodeProto]) -> list[NodeProto]:
    """Filter Gemm nodes that can be fused (no multiple pre/post connections).

    :param nodes: All nodes in the graph
    :return: List of fusable Gemm nodes
    """
    gemm_nodes = [node for node in nodes if node.op_type == "Gemm"]
    fusable_nodes = []
    for gemm_node in gemm_nodes:
        n_pre, n_post = _count_node_connections(gemm_node, nodes)
        if not (n_pre > 1 or n_post > 1):
            fusable_nodes.append(gemm_node)
    return fusable_nodes[::-1]


def _group_adjacent_gemm_nodes(
    gemm_nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[list[NodeProto]]:
    """Group adjacent Gemm nodes that can be fused together.

    :param gemm_nodes: List of Gemm nodes (in reverse order)
    :param initializers: Dictionary of initializers
    :return: List of Gemm node groups
    """
    chosen_output_names = [node.output[0] for node in gemm_nodes]
    fused_groups = []
    group_index = 0
    group = []

    while group_index < len(gemm_nodes):
        node = gemm_nodes[group_index]
        group_index += 1

        if node.input[1] not in initializers:
            raise ValueError(
                f"Gemm node {node.name} weight input {node.input[1]} not found in initializers."
            )

        group.append(node)
        if node.input[0] in chosen_output_names:
            continue

        if len(group) > 1:
            fused_groups.append(group[::-1])
        group = []

    fused_groups.reverse()
    return fused_groups


def _fuse_gemm_group(group: list[NodeProto], initializers: dict[str, TensorProto]) -> NodeProto:
    """Fuse a group of Gemm nodes into a single Gemm node.

    :param group: List of Gemm nodes to fuse
    :param initializers: Dictionary of initializers
    :return: Fused Gemm node
    """
    all_weights = []
    all_biases = []

    for gemm_node in group:
        if gemm_node.input[2] not in initializers:
            raise ValueError(
                f"Gemm node {gemm_node.name} bias input {gemm_node.input[2]} "
                "not found in initializers."
            )
        alpha, beta, trans_a, trans_b, weight, bias = _get_gemm_params(
            gemm_node, initializers, remove_initializers=True
        )
        if alpha != 1 or beta != 1:
            raise ValueError(
                f"Gemm node {gemm_node.name} has unsupported alpha={alpha} or beta={beta}. "
                "Only alpha=beta=1 is supported for fusion."
            )
        if trans_a != 0 or trans_b != 0:
            raise ValueError(
                f"Gemm node {gemm_node.name} has unsupported transA={trans_a} "
                f"or transB={trans_b}. Only transA=transB=0 is supported for fusion."
            )
        all_weights.append(weight)
        all_biases.append(bias)

    # Merge all weights and biases
    new_weight = all_weights[0]
    new_bias = all_biases[0]
    for i in range(1, len(all_weights)):
        new_weight = new_weight @ all_weights[i]
        new_bias = new_bias @ all_weights[i] + all_biases[i]

    # Create new initializers
    new_weight_name = group[0].input[1]
    new_bias_name = group[0].input[2]
    new_weight_tensor = onnx.numpy_helper.from_array(new_weight, new_weight_name)
    new_bias_tensor = onnx.numpy_helper.from_array(new_bias, new_bias_name)
    initializers[new_weight_name] = new_weight_tensor
    initializers[new_bias_name] = new_bias_tensor

    # Create fused node
    new_gemm_node = onnx.NodeProto()
    new_gemm_node.CopyFrom(group[0])
    new_gemm_node.ClearField("input")
    new_gemm_node.ClearField("output")
    new_gemm_node.input.extend([group[0].input[0], new_weight_name, new_bias_name])
    new_gemm_node.output.extend(group[-1].output)
    new_gemm_node.name = group[-1].name

    return new_gemm_node


def _fuse_gemm_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Fuse adjacent Gemm nodes into single Gemm operations."""
    # Filter and group Gemm nodes
    fusable_gemm_nodes = _filter_fusable_gemm_nodes(nodes)
    fused_groups = _group_adjacent_gemm_nodes(fusable_gemm_nodes, initializers)

    # Build lookup structures
    fused_output_names = [node.output[0] for group in fused_groups for node in group]
    group_start_map = {group[0].output[0]: i for i, group in enumerate(fused_groups)}

    # Replace nodes
    new_nodes = []
    for node in nodes:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        if node.output[0] not in fused_output_names:
            new_nodes.append(node)
            continue

        if node.output[0] not in group_start_map:
            continue

        group = fused_groups[group_start_map[node.output[0]]]
        fused_node = _fuse_gemm_group(group, initializers)
        new_nodes.append(fused_node)

    return new_nodes
