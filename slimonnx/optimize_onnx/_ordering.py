__docformat__ = "restructuredtext"
__all__ = ["_reorder_by_strict_topological_order"]

from onnx import NodeProto

from slimonnx.utils import get_next_nodes_mapping


def _reorder_by_strict_topological_order(nodes: list[NodeProto]) -> list[NodeProto]:
    next_nodes_mapping = get_next_nodes_mapping(nodes)

    # Topological sort
    visited = {node.name: False for node in nodes}
    stack = []

    def _topological_sort(node_name: str):
        nonlocal visited
        nonlocal next_nodes_mapping
        nonlocal stack
        visited[node_name] = True
        for next_node_name in next_nodes_mapping[node_name]:
            if not visited[next_node_name]:
                _topological_sort(next_node_name)
        stack.append(node_name)

    for node in nodes:
        if not visited[node.name]:
            _topological_sort(node.name)
    stack.reverse()

    # Reorder the nodes
    name_node_mapping = {node.name: node for node in nodes}
    new_nodes = [name_node_mapping[node_name] for node_name in stack]

    if len(nodes) != len(new_nodes):
        raise RuntimeError(
            f"Node count mismatch after topological ordering: "
            f"original={len(nodes)}, reordered={len(new_nodes)}."
        )

    return new_nodes
