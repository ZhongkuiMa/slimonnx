"""Transpose + BatchNorm + Transpose fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_transpose_bn_transpose"]

from onnx import NodeProto, TensorProto


def _is_consecutive_nodes(
    first_node: NodeProto, second_node: NodeProto, nodes: list[NodeProto]
) -> bool:
    """Check if second_node immediately follows first_node with no other consumers."""
    if first_node.output[0] != second_node.input[0]:
        return False
    first_output = first_node.output[0]
    consumer_count = sum(1 for node in nodes if first_output in node.input)
    return consumer_count == 1


def _get_perm_attribute(node: NodeProto) -> tuple[int, ...] | None:
    """Extract perm attribute from Transpose node."""
    for attr in node.attribute:
        if attr.name == "perm":
            return tuple(attr.ints)
    return None


def detect_transpose_bn_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect Transpose + BatchNormalization + Transpose fusion pattern.

    Pattern: Transpose(0,2,1) -> BatchNormalization -> Transpose(0,2,1)
    Can be fused into single Gemm operation.
    Requires both Transpose operations to have perm=(0,2,1).

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 2):
        transpose1_node = nodes[i]
        bn_node = nodes[i + 1]
        transpose2_node = nodes[i + 2]

        # Check pattern: Transpose -> BatchNormalization -> Transpose
        if (
            transpose1_node.op_type != "Transpose"
            or bn_node.op_type != "BatchNormalization"
            or transpose2_node.op_type != "Transpose"
        ):
            continue

        # Check consecutive connections
        if not _is_consecutive_nodes(transpose1_node, bn_node, nodes):
            continue
        if not _is_consecutive_nodes(bn_node, transpose2_node, nodes):
            continue

        # Check if both Transpose nodes have perm=(0,2,1)
        perm1 = _get_perm_attribute(transpose1_node)
        perm2 = _get_perm_attribute(transpose2_node)

        if perm1 != (0, 2, 1) or perm2 != (0, 2, 1):
            continue

        # Check if BN has all required parameters
        bn_inputs = bn_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        instances.append(
            {
                "transpose1_node": transpose1_node.name,
                "bn_node": bn_node.name,
                "transpose2_node": transpose2_node.name,
                "bn_scale": bn_inputs[1],
                "perm": perm1,
                "can_fuse": True,
            }
        )

    return instances
