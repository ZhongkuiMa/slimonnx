"""Gemm + BatchNorm fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_gemm_reshape_bn",
    "detect_bn_reshape_gemm",
    "detect_bn_gemm",
]

from onnx import NodeProto, TensorProto


def _is_consecutive_nodes(
    first_node: NodeProto, second_node: NodeProto, nodes: list[NodeProto]
) -> bool:
    """Check if second_node immediately follows first_node with no other consumers.

    :param first_node: First node in pattern
    :param second_node: Second node in pattern
    :param nodes: All nodes in graph
    :return: True if nodes are consecutive with no branching
    """
    if first_node.output[0] != second_node.input[0]:
        return False

    first_output = first_node.output[0]
    consumer_count = sum(1 for node in nodes if first_output in node.input)

    return consumer_count == 1


def detect_gemm_reshape_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect Gemm + Reshape + BatchNormalization fusion pattern.

    Pattern: Gemm -> Reshape -> BatchNormalization
    Can be fused into Gemm + Reshape with modified Gemm weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 2):
        gemm_node = nodes[i]
        reshape_node = nodes[i + 1]
        bn_node = nodes[i + 2]

        # Check pattern: Gemm -> Reshape -> BatchNormalization
        if (
            gemm_node.op_type != "Gemm"
            or reshape_node.op_type != "Reshape"
            or bn_node.op_type != "BatchNormalization"
        ):
            continue

        # Check consecutive connections
        if not _is_consecutive_nodes(gemm_node, reshape_node, nodes):
            continue
        if not _is_consecutive_nodes(reshape_node, bn_node, nodes):
            continue

        # Check if Gemm has constant weight
        if len(gemm_node.input) < 2 or gemm_node.input[1] not in initializers:
            continue

        # Check if Reshape has constant shape
        if len(reshape_node.input) < 2 or reshape_node.input[1] not in initializers:
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
                "gemm_node": gemm_node.name,
                "reshape_node": reshape_node.name,
                "bn_node": bn_node.name,
                "gemm_weight": gemm_node.input[1],
                "reshape_shape": reshape_node.input[1],
                "bn_scale": bn_inputs[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_reshape_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + Reshape + Gemm fusion pattern.

    Pattern: BatchNormalization -> Reshape -> Gemm
    Can be fused into Reshape + Gemm with modified Gemm weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 2):
        bn_node = nodes[i]
        reshape_node = nodes[i + 1]
        gemm_node = nodes[i + 2]

        # Check pattern: BatchNormalization -> Reshape -> Gemm
        if (
            bn_node.op_type != "BatchNormalization"
            or reshape_node.op_type != "Reshape"
            or gemm_node.op_type != "Gemm"
        ):
            continue

        # Check consecutive connections
        if not _is_consecutive_nodes(bn_node, reshape_node, nodes):
            continue
        if not _is_consecutive_nodes(reshape_node, gemm_node, nodes):
            continue

        # Check if BN has all required parameters
        bn_inputs = bn_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        # Check if Reshape has constant shape
        if len(reshape_node.input) < 2 or reshape_node.input[1] not in initializers:
            continue

        # Check if Gemm has constant weight
        if len(gemm_node.input) < 2 or gemm_node.input[1] not in initializers:
            continue

        instances.append(
            {
                "bn_node": bn_node.name,
                "reshape_node": reshape_node.name,
                "gemm_node": gemm_node.name,
                "bn_scale": bn_inputs[1],
                "reshape_shape": reshape_node.input[1],
                "gemm_weight": gemm_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + Gemm fusion pattern.

    Pattern: BatchNormalization -> Gemm
    Can be fused into single Gemm with modified weights/biases.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of pattern instances
    """
    instances = []

    for i in range(len(nodes) - 1):
        bn_node = nodes[i]
        gemm_node = nodes[i + 1]

        # Check pattern: BatchNormalization -> Gemm
        if bn_node.op_type != "BatchNormalization" or gemm_node.op_type != "Gemm":
            continue

        # Check consecutive connection
        if not _is_consecutive_nodes(bn_node, gemm_node, nodes):
            continue

        # Check if BN has all required parameters
        bn_inputs = bn_node.input
        if len(bn_inputs) < 5:
            continue

        required_params = [bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]]
        if not all(param in initializers for param in required_params):
            continue

        # Check if Gemm has constant weight
        if len(gemm_node.input) < 2 or gemm_node.input[1] not in initializers:
            continue

        instances.append(
            {
                "bn_node": bn_node.name,
                "gemm_node": gemm_node.name,
                "bn_scale": bn_inputs[1],
                "gemm_weight": gemm_node.input[1],
                "can_fuse": True,
            }
        )

    return instances
