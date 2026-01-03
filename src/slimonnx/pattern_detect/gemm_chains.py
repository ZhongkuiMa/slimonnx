"""Gemm chain fusion pattern detection (linear operation merging)."""

__docformat__ = "restructuredtext"
__all__ = ["detect_gemm_gemm"]

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect.utils import (
    has_constant_weight,
    is_consecutive_nodes,
)


def _get_gemm_attributes(node: NodeProto) -> dict[str, float | int]:
    """Extract alpha, beta, transA, transB attributes from Gemm node."""
    attrs = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}

    for attr in node.attribute:
        if attr.name in attrs:
            if attr.name in ["alpha", "beta"]:
                attrs[attr.name] = attr.f
            else:  # transA, transB
                attrs[attr.name] = attr.i

    return attrs


def detect_gemm_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect consecutive Gemm operations that can be merged (linear operation merging).

    Pattern: Gemm -> Gemm
    Two consecutive Gemm operations can be merged into single Gemm.
    This is a form of linear operation merging:
    (X @ W1 + b1) @ W2 + b2 = X @ (W1 @ W2) + (b1 @ W2 + b2)

    Requirements:
    - Both Gemm nodes must have alpha=1, beta=1
    - Both must have transA=0, transB=0
    - Both must have constant weights

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 1):
        gemm1_node = nodes[i]
        gemm2_node = nodes[i + 1]

        # Check pattern: Gemm -> Gemm
        if gemm1_node.op_type != "Gemm" or gemm2_node.op_type != "Gemm":
            continue

        # Check consecutive connection
        if not is_consecutive_nodes(gemm1_node, gemm2_node, nodes):
            continue

        # Check if both Gemm nodes have constant weights
        if not has_constant_weight(gemm1_node, initializers):
            continue
        if not has_constant_weight(gemm2_node, initializers):
            continue

        # Get attributes
        attrs1 = _get_gemm_attributes(gemm1_node)
        attrs2 = _get_gemm_attributes(gemm2_node)

        # Check if fusion is possible (requires alpha=1, beta=1, transA=0, transB=0)
        can_fuse = (
            attrs1["alpha"] == 1.0
            and attrs1["beta"] == 1.0
            and attrs1["transA"] == 0
            and attrs1["transB"] == 0
            and attrs2["alpha"] == 1.0
            and attrs2["beta"] == 1.0
            and attrs2["transA"] == 0
            and attrs2["transB"] == 0
        )

        instances.append(
            {
                "gemm1_node": gemm1_node.name,
                "gemm2_node": gemm2_node.name,
                "gemm1_weight": gemm1_node.input[1],
                "gemm2_weight": gemm2_node.input[1],
                "can_fuse": can_fuse,
                "attrs1": attrs1,
                "attrs2": attrs2,
            }
        )

    return instances
