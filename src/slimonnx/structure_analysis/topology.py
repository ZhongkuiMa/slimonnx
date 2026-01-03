"""ONNX model topology utilities."""

__docformat__ = "restructuredtext"
__all__ = ["build_topology", "export_topology_json"]

import json
from pathlib import Path
from typing import Any

from onnx import NodeProto


def build_topology(nodes: list[NodeProto]) -> dict:
    """Build topological graph from nodes.

    :param nodes: Model nodes
    :return: Topology dictionary with predecessors/successors
    """
    # Build output-to-node mapping
    output_to_node = {}
    for node in nodes:
        for out in node.output:
            output_to_node[out] = node.name if node.name else f"{node.op_type}_unnamed"

    # Build predecessor/successor relationships
    topology = {}
    for node in nodes:
        node_name = node.name if node.name else f"{node.op_type}_unnamed"

        predecessors = [output_to_node[inp] for inp in node.input if inp in output_to_node]

        topology[node_name] = {
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "predecessors": predecessors,
        }

    # Add successors
    for info in topology.values():
        successors = [
            other_name
            for out in info["outputs"]
            for other_name, other_info in topology.items()
            if out in other_info["inputs"]
        ]
        info["successors"] = successors

    return topology


def export_topology_json(
    nodes: list[NodeProto],
    output_path: str,
    data_shapes: dict[str, int | list[int]] | None = None,
) -> None:
    """Export topology as JSON for baselines.

    :param nodes: Model nodes
    :param output_path: JSON output path
    :param data_shapes: Optional shape information
    """
    # Build node list
    node_list = []
    for node in nodes:
        node_name = node.name if node.name else f"{node.op_type}_unnamed"
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
        node_name = node.name if node.name else f"{node.op_type}_unnamed"
        for out in node.output:
            output_to_node[out] = node_name

    edges = [
        {
            "from": output_to_node[inp],
            "to": node.name if node.name else f"{node.op_type}_unnamed",
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
