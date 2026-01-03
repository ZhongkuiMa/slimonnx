"""Gemm + BatchNorm fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_bn_gemm",
    "detect_bn_reshape_gemm",
    "detect_gemm_reshape_bn",
]

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect.utils import (
    has_constant_weight,
    is_consecutive_nodes,
    validate_bn_inputs,
)


def detect_gemm_reshape_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(gemm_node, reshape_node, nodes):
            continue
        if not is_consecutive_nodes(reshape_node, bn_node, nodes):
            continue

        # Check if Gemm has constant weight
        if not has_constant_weight(gemm_node, initializers):
            continue

        # Check if Reshape has constant shape
        if not has_constant_weight(reshape_node, initializers):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(bn_node, initializers):
            continue

        instances.append(
            {
                "gemm_node": gemm_node.name,
                "reshape_node": reshape_node.name,
                "bn_node": bn_node.name,
                "gemm_weight": gemm_node.input[1],
                "reshape_shape": reshape_node.input[1],
                "bn_scale": bn_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_reshape_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(bn_node, reshape_node, nodes):
            continue
        if not is_consecutive_nodes(reshape_node, gemm_node, nodes):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(bn_node, initializers):
            continue

        # Check if Reshape has constant shape
        if not has_constant_weight(reshape_node, initializers):
            continue

        # Check if Gemm has constant weight
        if not has_constant_weight(gemm_node, initializers):
            continue

        instances.append(
            {
                "bn_node": bn_node.name,
                "reshape_node": reshape_node.name,
                "gemm_node": gemm_node.name,
                "bn_scale": bn_node.input[1],
                "reshape_shape": reshape_node.input[1],
                "gemm_weight": gemm_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_bn_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
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
        if not is_consecutive_nodes(bn_node, gemm_node, nodes):
            continue

        # Check if BN has all required parameters
        if not validate_bn_inputs(bn_node, initializers):
            continue

        # Check if Gemm has constant weight
        if not has_constant_weight(gemm_node, initializers):
            continue

        instances.append(
            {
                "bn_node": bn_node.name,
                "gemm_node": gemm_node.name,
                "bn_scale": bn_node.input[1],
                "gemm_weight": gemm_node.input[1],
                "can_fuse": True,
            }
        )

    return instances
