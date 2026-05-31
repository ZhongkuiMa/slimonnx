"""Depthwise convolution fusion optimizations."""

__docformat__ = "restructuredtext"
__all__ = ["_fuse_depthwise_conv_bn_or_bn_depthwise_conv"]

import numpy as np
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._bn_conv import _fuse_conv_bn_generic
from slimonnx.optimize_onnx._utils import _get_conv_params


def _get_conv_group_attr(node: NodeProto) -> int:
    """Extract group attribute from Conv node.

    :param node: Conv node.

    :return: Group value (default 1)
    """
    for attr in node.attribute:
        if attr.name == "group":
            return int(attr.i)
    return 1


def _is_depthwise_conv(node: NodeProto, initializers: dict[str, TensorProto]) -> bool:
    """Check if a Conv node is depthwise convolution.

    Depthwise convolution: group == in_channels == out_channels
    Conv weight shape: [out_channels, in_channels/group, kH, kW]
    For depthwise: in_channels/group == 1 and group == out_channels

    :param node: Conv node to check.

    :param initializers: Model initializers.

    :return: True if depthwise convolution
    """
    if node.op_type != "Conv":
        return False

    group = _get_conv_group_attr(node)
    if group == 1:
        return False

    if len(node.input) < 2 or node.input[1] not in initializers:
        return False

    weight_tensor = initializers[node.input[1]]
    weight_shape = [int(d) for d in weight_tensor.dims]

    out_channels = weight_shape[0]
    in_channels_per_group = weight_shape[1]

    return bool(in_channels_per_group == 1 and group == out_channels)


def _depthwise_simplified_bias(
    _weight: np.ndarray,
    bias: np.ndarray,
    _bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    _weight_axis_bn_to_op: tuple[int, ...],
    _bias_reduce_axes: tuple[int, ...],
) -> np.ndarray:
    """Return the simplified depthwise BN->Conv bias: ``bias + bn_bias``.

    For depthwise convolution every output channel is computed from a
    single input channel with its own ``(1, kH, kW)`` filter, so the
    generic ``bias + sum_{spatial}(W * beta)`` term collapses to a
    per-channel scalar ``bias + bn_bias`` (legacy historical formula --
    not the mathematically exact closed form in general).
    """
    return bias + bn_bias


def _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
) -> list[NodeProto]:
    """Fuse depthwise Conv + BatchNormalization or BatchNormalization + depthwise Conv.

    For depthwise convolution with group=C (channels):
    - Conv weight shape: ``[C, 1, kH, kW]``
    - Each channel has its own ``1xkHxkW`` filter

    Fusion formulae:
    - DepthwiseConv->BN: ``new_weight = weight * gamma.reshape(-1, 1, 1, 1)``;
      ``new_bias = bias * gamma + beta``.
    - BN->DepthwiseConv: same ``new_weight``; ``new_bias = bias + beta``
      (simplified per-channel form; see ``_depthwise_simplified_bias``).

    :param nodes: List of nodes.

    :param initializers: Dictionary of initializers (mutated).

    :param is_conv_bn: True for Conv+BN, False for BN+Conv.

    :return: Optimized nodes
    """
    return _fuse_conv_bn_generic(
        nodes,
        initializers,
        match_linear_node=_is_depthwise_conv,
        get_linear_params=_get_conv_params,
        conv_first=is_conv_bn,
        weight_axis_op_to_bn=(-1, 1, 1, 1),
        weight_axis_bn_to_op=(-1, 1, 1, 1),
        bias_reduce_axes=(),
        skip_padded=False,
        bias_bn_to_op_fn=_depthwise_simplified_bias,
    )
