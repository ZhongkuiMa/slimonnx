"""ONNX model topology utilities."""

__docformat__ = "restructuredtext"
__all__ = ["build_topology", "export_topology_json"]

import json
from pathlib import Path
from typing import Any

from onnx import NodeProto


def build_topology(nodes: list[NodeProto]) -> dict:
    """Build topological graph from nodes.

    Runs in O(n + e) time -- one pass to map outputs to producer names, one
    pass to populate predecessors / successors via that map.

    :param nodes: Model nodes.

    :return: Topology dictionary with predecessors/successors
    """
    # Pass 1: stable node names + output -> producer-name index.
    node_names: list[str] = [node.name or f"{node.op_type}_unnamed" for node in nodes]
    output_to_node: dict[str, str] = {}
    for node, name in zip(nodes, node_names, strict=True):
        for out in node.output:
            output_to_node[out] = name

    # Pass 2: build the topology table with empty successor slots.
    topology: dict[str, dict] = {
        name: {
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "predecessors": [output_to_node[inp] for inp in node.input if inp in output_to_node],
            "successors": [],
        }
        for node, name in zip(nodes, node_names, strict=True)
    }

    # Pass 3: invert predecessors to successors. Linear in total edges, no
    # nested scan -- replaces the previous O(n^2) "look at every other node".
    # A consumer that lists the same predecessor twice (e.g. ``Add(Y, Y)``)
    # appears only once on the producer's successor list, matching the
    # original O(n^2) behaviour which dedup'd implicitly.
    for name, info in topology.items():
        seen: set[str] = set()
        for predecessor in info["predecessors"]:
            if predecessor in seen:
                continue
            seen.add(predecessor)
            topology[predecessor]["successors"].append(name)

    return topology


def export_topology_json(
    nodes: list[NodeProto],
    output_path: str,
    data_shapes: dict[str, int | list[int]] | None = None,
) -> None:
    """Export topology as JSON for baselines.

    :param nodes: Model nodes.

    :param output_path: JSON output path.

    :param data_shapes: Optional shape information.

    """
    # Build node list
    node_list = []
    for node in nodes:
        node_name = node.name or f"{node.op_type}_unnamed"
        node_info: dict[str, Any] = {
            "name": node_name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }

        # Add shapes if available
        if data_shapes is not None:
            shapes = {}
            for out in node.output:
                if out in data_shapes:
                    shapes[out] = data_shapes[out]
            if shapes:
                node_info["shapes"] = shapes

        node_list.append(node_info)

    # Build edge list
    output_to_node = {}
    for node in nodes:
        node_name = node.name or f"{node.op_type}_unnamed"
        for out in node.output:
            output_to_node[out] = node_name

    edges = [
        {
            "from": output_to_node[inp],
            "to": node.name or f"{node.op_type}_unnamed",
            "tensor": inp,
        }
        for node in nodes
        for inp in node.input
        if inp in output_to_node
    ]

    # Create JSON structure
    topology_json = {
        "nodes": node_list,
        "edges": edges,
        "node_count": len(node_list),
        "edge_count": len(edges),
    }

    # Write to file
    with Path(output_path).open("w") as f:
        json.dump(topology_json, f, indent=2)
