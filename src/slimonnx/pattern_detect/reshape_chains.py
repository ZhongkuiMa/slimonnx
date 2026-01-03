"""Reshape chain detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_consecutive_reshape"]

from onnx import NodeProto


def detect_consecutive_reshape(nodes: list[NodeProto]) -> list[dict]:
    """Detect consecutive Reshape operations.

    :param nodes: Model nodes
    :return: List of detected pattern instances
    """
    # Build output-to-node mapping
    output_to_node = {}
    for node in nodes:
        for out in node.output:
            output_to_node[out] = node

    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Reshape":
            continue

        # Check if input comes from another Reshape
        if len(node.input) > 0:
            input_name = node.input[0]
            if input_name in output_to_node:
                prev_node = output_to_node[input_name]
                if prev_node.op_type == "Reshape":
                    instances.append(
                        {
                            "first_node": (
                                prev_node.name if prev_node.name else f"Reshape_{i - 1}"
                            ),
                            "second_node": node.name if node.name else f"Reshape_{i}",
                        }
                    )

    return instances
