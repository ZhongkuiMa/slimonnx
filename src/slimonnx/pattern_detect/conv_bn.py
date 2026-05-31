"""Conv + BatchNorm fusion pattern detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_bn_conv",
    "detect_bn_conv_transpose",
    "detect_conv_bn",
    "detect_conv_transpose_bn",
]

from onnx import NodeProto, TensorProto

from slimonnx.pattern_detect.utils import (
    has_constant_weight,
    is_consecutive_nodes,
    validate_bn_inputs,
)


def _detect_conv_family_bn_pair(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    conv_op: str,
    *,
    conv_first: bool,
    conv_key: str,
    conv_weight_key: str,
) -> list[dict]:
    """Detect a Conv-family/BatchNormalization directed pair.

    Shared scanner for the four ``conv_bn`` variants. Walks the node list
    looking for consecutive single-consumer pairs of the requested op
    types, validates that the Conv-family node has a constant weight and
    the BN node has its full parameter set, and yields one match dict
    per location.

    :param nodes: Model nodes.

    :param initializers: Initializer map keyed by name.

    :param conv_op: ``"Conv"`` or ``"ConvTranspose"`` -- the op_type to
        pair with BatchNormalization.

    :param conv_first: ``True`` for ``Conv -> BN``, ``False`` for
        ``BN -> Conv``. Determines which slot validates which constraints.

    :param conv_key: Key under which the Conv-family node name lands in
        the result dict (``"conv_node"`` / ``"conv_transpose_node"``).

    :param conv_weight_key: Key under which the Conv-family weight tensor
        name lands in the result dict.

    :return: List of match dicts with the same legacy schema the four
        public detectors used to return.
    """
    instances: list[dict] = []
    bn_op = "BatchNormalization"
    pre_op, post_op = (conv_op, bn_op) if conv_first else (bn_op, conv_op)

    for i in range(len(nodes) - 1):
        curr_node = nodes[i]
        next_node = nodes[i + 1]

        if curr_node.op_type != pre_op or next_node.op_type != post_op:
            continue
        if not is_consecutive_nodes(curr_node, next_node, nodes):
            continue

        conv_node = curr_node if conv_first else next_node
        bn_node = next_node if conv_first else curr_node

        if not has_constant_weight(conv_node, initializers):
            continue
        if not validate_bn_inputs(bn_node, initializers):
            continue

        instances.append(
            {
                conv_key: conv_node.name,
                "bn_node": bn_node.name,
                conv_weight_key: conv_node.input[1],
                "bn_scale": bn_node.input[1],
                "can_fuse": True,
            }
        )

    return instances


def detect_conv_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect Conv + BatchNormalization fusion pattern.

    Pattern: Conv -> BatchNormalization
    Can be fused into single Conv with modified weights/biases.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of initializers.

    :param data_shapes: Optional shape information (unused).

    :return: List of pattern instances with Conv and BN node info.
    """
    del data_shapes
    return _detect_conv_family_bn_pair(
        nodes,
        initializers,
        conv_op="Conv",
        conv_first=True,
        conv_key="conv_node",
        conv_weight_key="conv_weight",
    )


def detect_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + Conv fusion pattern.

    Pattern: BatchNormalization -> Conv
    Can be fused into single Conv with modified weights/biases.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of initializers.

    :param data_shapes: Optional shape information (unused).

    :return: List of pattern instances with BN and Conv node info.
    """
    del data_shapes
    return _detect_conv_family_bn_pair(
        nodes,
        initializers,
        conv_op="Conv",
        conv_first=False,
        conv_key="conv_node",
        conv_weight_key="conv_weight",
    )


def detect_conv_transpose_bn(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect ConvTranspose + BatchNormalization fusion pattern.

    Pattern: ConvTranspose -> BatchNormalization
    Can be fused into single ConvTranspose with modified weights/biases.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of initializers.

    :param data_shapes: Optional shape information (unused).

    :return: List of pattern instances with ConvTranspose and BN node info.
    """
    del data_shapes
    return _detect_conv_family_bn_pair(
        nodes,
        initializers,
        conv_op="ConvTranspose",
        conv_first=True,
        conv_key="conv_transpose_node",
        conv_weight_key="conv_transpose_weight",
    )


def detect_bn_conv_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect BatchNormalization + ConvTranspose fusion pattern.

    Pattern: BatchNormalization -> ConvTranspose
    Can be fused into single ConvTranspose with modified weights/biases.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of initializers.

    :param data_shapes: Optional shape information (unused).

    :return: List of pattern instances with BN and ConvTranspose node info.
    """
    del data_shapes
    return _detect_conv_family_bn_pair(
        nodes,
        initializers,
        conv_op="ConvTranspose",
        conv_first=False,
        conv_key="conv_transpose_node",
        conv_weight_key="conv_transpose_weight",
    )
