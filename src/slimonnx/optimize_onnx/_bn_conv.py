"""Fuse BatchNormalization with Conv and ConvTranspose operators."""

__docformat__ = "restructuredtext"
__all__ = [
    "_fuse_conv_bn_generic",
    "_fuse_conv_bn_or_bn_conv",
    "_fuse_conv_transpose_bn_or_bn_conv_transpose",
]

from collections.abc import Callable

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import (
    _get_batchnorm_params,
    _get_conv_params,
    _get_conv_transpose_params,
    _is_only_next_node,
    compute_batchnorm_fusion_params,
)

# Type alias for the conv-family parameter getters: returns
# ``(weight, bias, attrs)`` with ``remove_initializers`` controlling whether
# the read also pops the source initializer entries.
_LinearParamGetter = Callable[
    [NodeProto, dict[str, TensorProto], bool],
    tuple[np.ndarray, np.ndarray, dict],
]


_LinearMatchFn = Callable[[NodeProto, dict[str, TensorProto]], bool]
_BiasFormulaFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]],
    np.ndarray,
]


def _generic_bn_to_op_bias(
    weight: np.ndarray,
    bias: np.ndarray,
    _bn_weight: np.ndarray,
    bn_bias: np.ndarray,
    weight_axis_bn_to_op: tuple[int, ...],
    bias_reduce_axes: tuple[int, ...],
) -> np.ndarray:
    """Compute the BN -> op bias term for a generic linear operator.

    Returns ``bias + sum_{spatial}(W * beta)`` with ``beta`` broadcast on
    the linear op's input channel axis. This is the math-correct closed
    form when the receptive field does no implicit padding.
    """
    return bias + np.sum(
        weight * bn_bias.reshape(weight_axis_bn_to_op),
        axis=bias_reduce_axes,
    )


def _fuse_conv_bn_generic(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    *,
    conv_op_type: str | None = None,
    match_linear_node: _LinearMatchFn | None = None,
    get_linear_params: _LinearParamGetter,
    conv_first: bool,
    weight_axis_op_to_bn: tuple[int, ...],
    weight_axis_bn_to_op: tuple[int, ...],
    bias_reduce_axes: tuple[int, ...],
    skip_padded: bool,
    bias_bn_to_op_fn: _BiasFormulaFn = _generic_bn_to_op_bias,
) -> list[NodeProto]:
    """Fuse a BN/conv-family op pair in either direction.

    Shared scanner for the four Conv / ConvTranspose x (op_to_bn / bn_to_op)
    fusion variants. The reshape and reduce axes depend only on which axis
    of the linear op's weight tensor carries the channel dimension, so the
    arithmetic is captured by ``weight_axis_*`` plus ``bias_reduce_axes``.

    :param nodes: Model nodes.

    :param initializers: Initializer map keyed by tensor name. Mutated:
        BN parameter initializers and the conv op's weight/bias are
        popped, replaced by fused weight/bias.

    :param conv_op_type: ``"Conv"`` or ``"ConvTranspose"`` -- used when the
        conv slot can be identified by ``op_type`` equality. Mutually
        exclusive with ``match_linear_node``.

    :param match_linear_node: Predicate ``(node, initializers) -> bool`` that
        decides whether a node is a valid conv slot. Used by the
        depthwise variant which needs to inspect the weight shape, not just
        ``op_type``. Mutually exclusive with ``conv_op_type``.

    :param get_linear_params: Function to extract ``(weight, bias, attrs)``
        from a linear-op NodeProto.

    :param conv_first: ``True`` for ``LinearOp -> BN``, ``False`` for
        ``BN -> LinearOp``.

    :param weight_axis_op_to_bn: ``shape`` tuple for ``bn_weight.reshape``
        when fusing op_to_bn (downstream BN consumes op's output channel).

    :param weight_axis_bn_to_op: ``shape`` tuple for ``bn_weight.reshape``
        when fusing bn_to_op (upstream BN feeds op's input channel).

    :param bias_reduce_axes: Axes to ``np.sum`` over when collapsing the
        ``W * beta`` term into the downstream bias on the bn_to_op path.

    :param skip_padded: If ``True`` and the bn_to_op direction is being
        applied, skip fusion when the linear op has non-zero padding -
        the math is invalid when ``bn_bias != 0`` and the conv pads zeros.

    :param bias_bn_to_op_fn: Bias-formula override for the BN -> op
        direction. Defaults to the generic ``bias + sum(W * beta)`` closed
        form. The depthwise variant supplies a simplified
        ``bias + bn_bias`` that exploits the per-channel-independent
        structure.

    :return: New node list with matched pairs fused into a single linear op.
    """
    if (conv_op_type is None) == (match_linear_node is None):
        raise ValueError(
            "_fuse_conv_bn_generic requires exactly one of conv_op_type or match_linear_node"
        )

    new_nodes: list[NodeProto] = []
    bn_op = "BatchNormalization"

    pre_node: NodeProto | None = None
    for node in nodes:
        new_nodes.append(node)

        if pre_node is None or not _is_only_next_node(pre_node, node, nodes):
            pre_node = node
            continue

        # Identify which slot holds the linear op and which holds the BN.
        if conv_first:
            linear_candidate, bn_candidate = pre_node, node
        else:
            linear_candidate, bn_candidate = node, pre_node

        if bn_candidate.op_type != bn_op:
            pre_node = node
            continue
        if conv_op_type is not None:
            if linear_candidate.op_type != conv_op_type:
                pre_node = node
                continue
        elif not match_linear_node(linear_candidate, initializers):  # type: ignore[misc]
            pre_node = node
            continue

        conv_node, bn_node = linear_candidate, bn_candidate

        # Padding-skip check is dry-run: read attrs without popping initializers
        # so a skip leaves the graph untouched.
        if skip_padded and not conv_first:
            # Dry-run: third arg controls remove_initializers; False = read only.
            _, _, attrs = get_linear_params(conv_node, initializers, False)  # noqa: FBT003
            if any(p != 0 for p in attrs["pads"]):
                pre_node = node
                continue

        # Pop the last two nodes; we are committed to fusion now.
        new_nodes.pop()
        new_nodes.pop()

        epsilon, scale, bn_param_bias, mean, var = _get_batchnorm_params(
            bn_node, initializers, remove_initializers=True
        )
        # Commit: third arg controls remove_initializers; True = pop on read.
        weight, bias, _attrs = get_linear_params(conv_node, initializers, True)  # noqa: FBT003

        target_dtype = weight.dtype
        bn_weight, bn_bias = compute_batchnorm_fusion_params(
            epsilon, scale, bn_param_bias, mean, var, target_dtype
        )

        if conv_first:
            new_weight = (weight * bn_weight.reshape(weight_axis_op_to_bn)).astype(
                target_dtype, copy=False
            )
            new_bias = (bias * bn_weight + bn_bias).astype(target_dtype, copy=False)
        else:
            new_weight = (weight * bn_weight.reshape(weight_axis_bn_to_op)).astype(
                target_dtype, copy=False
            )
            new_bias = bias_bn_to_op_fn(
                weight,
                bias,
                bn_weight,
                bn_bias,
                weight_axis_bn_to_op,
                bias_reduce_axes,
            ).astype(target_dtype, copy=False)

        new_weight_name = conv_node.input[1]
        if len(conv_node.input) > 2:
            new_bias_name = conv_node.input[2]
        else:
            new_bias_name = conv_node.input[1] + "_bias"
        initializers[new_weight_name] = onnx.numpy_helper.from_array(new_weight, new_weight_name)
        initializers[new_bias_name] = onnx.numpy_helper.from_array(new_bias, new_bias_name)

        new_node = onnx.NodeProto()
        new_node.CopyFrom(conv_node)
        new_node.ClearField("input")
        new_node.ClearField("output")

        if conv_first:
            new_node.input.extend([conv_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(bn_node.output)
        else:
            new_node.input.extend([bn_node.input[0], new_weight_name, new_bias_name])
            new_node.output.extend(conv_node.output)

        new_nodes.append(new_node)
        pre_node = new_node

    return new_nodes


def _fuse_conv_bn_or_bn_conv(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_bn: bool = True,
) -> list[NodeProto]:
    """Fuse Conv+BN or BN+Conv pattern.

    Conv weight layout is ``(out_C, in_C/group, kH, kW)``; the channel axis
    is 0 on the output side and 1 on the input side. The BN->Conv direction
    is skipped on padded convs because the bias term would otherwise be
    applied outside the receptive field where zeros are padded.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of ONNX initializers (mutated).

    :param is_conv_bn: True for Conv->BN, False for BN->Conv.

    :return: New node list with fused pairs.
    """
    return _fuse_conv_bn_generic(
        nodes,
        initializers,
        conv_op_type="Conv",
        get_linear_params=_get_conv_params,
        conv_first=is_conv_bn,
        weight_axis_op_to_bn=(-1, 1, 1, 1),
        weight_axis_bn_to_op=(1, -1, 1, 1),
        bias_reduce_axes=(1, 2, 3),
        skip_padded=True,
    )


def _fuse_conv_transpose_bn_or_bn_conv_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    is_conv_transpose_bn: bool = True,
) -> list[NodeProto]:
    """Fuse ConvTranspose+BN or BN+ConvTranspose pattern.

    ConvTranspose weight layout is ``(in_C, out_C/group, kH, kW)``; the
    channel axes are swapped versus Conv. There is no padding-skip on
    ConvTranspose: legacy behaviour did not guard the bn_to_op direction
    and tests pin it.

    :param nodes: List of ONNX nodes.

    :param initializers: Dictionary of ONNX initializers (mutated).

    :param is_conv_transpose_bn: True for ConvTranspose->BN, False for
        BN->ConvTranspose.

    :return: New node list with fused pairs.
    """
    return _fuse_conv_bn_generic(
        nodes,
        initializers,
        conv_op_type="ConvTranspose",
        get_linear_params=_get_conv_transpose_params,
        conv_first=is_conv_transpose_bn,
        weight_axis_op_to_bn=(1, -1, 1, 1),
        weight_axis_bn_to_op=(-1, 1, 1, 1),
        bias_reduce_axes=(0, 2, 3),
        skip_padded=False,
    )
