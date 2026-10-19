"""Conservative canonicalization of ONNX linear quantization operators."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["_canonicalize_quantization"]

from collections import Counter

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper


def _attribute_int(node: NodeProto, name: str, default: int) -> int:
    """Return one integer node attribute without accepting other encodings."""
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _standard_domain(node: NodeProto) -> bool:
    """Return whether a node uses the standard ONNX operator domain."""
    return node.domain in {"", "ai.onnx"}


def _fold_static_dequantize(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> TensorProto | None:
    """Fold a fully static DequantizeLinear node, or decline conservatively."""
    if (
        not _standard_domain(node)
        or node.op_type != "DequantizeLinear"
        or len(node.input) not in {2, 3}
        or any(name not in initializers for name in node.input)
        or any(attribute.name == "block_size" for attribute in node.attribute)
    ):
        return None

    data = numpy_helper.to_array(initializers[node.input[0]])
    scale = numpy_helper.to_array(initializers[node.input[1]])
    if not np.issubdtype(data.dtype, np.integer) or not np.issubdtype(scale.dtype, np.floating):
        return None
    if not np.isfinite(scale).all() or not np.greater(scale, 0).all():
        return None

    if len(node.input) == 3:
        zero_point = numpy_helper.to_array(initializers[node.input[2]])
        if zero_point.dtype != data.dtype:
            return None
    else:
        zero_point = np.zeros_like(scale, dtype=data.dtype)

    if scale.size == 1 and zero_point.size == 1:
        # ONNX Runtime and the VNN-COMP reference frontend treat one-element
        # quantization parameters as per-tensor scalars even when exporters
        # encode one as ``[1]`` and the other as ``[]``.
        broadcast_scale = scale.reshape(())
        broadcast_zero_point = zero_point.reshape(())
    elif scale.ndim == 1:
        if zero_point.shape != scale.shape:
            return None
        axis = _attribute_int(node, "axis", 1)
        axis = axis + data.ndim if axis < 0 else axis
        if axis < 0 or axis >= data.ndim or scale.shape[0] != data.shape[axis]:
            return None
        broadcast_shape = [1] * data.ndim
        broadcast_shape[axis] = scale.shape[0]
        broadcast_scale = scale.reshape(broadcast_shape)
        broadcast_zero_point = zero_point.reshape(broadcast_shape)
    else:
        return None

    result = (data.astype(scale.dtype) - broadcast_zero_point.astype(scale.dtype)) * broadcast_scale
    return numpy_helper.from_array(np.asarray(result, dtype=scale.dtype), name=node.output[0])


def _eligible_qdq_pair(
    quantize: NodeProto,
    dequantize: NodeProto,
    initializers: dict[str, TensorProto],
    consumers: Counter[str],
    graph_outputs: set[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return scalar scale/zero-point values when a Q-DQ pair is exact to lower."""
    if (
        not _standard_domain(quantize)
        or not _standard_domain(dequantize)
        or quantize.op_type != "QuantizeLinear"
        or dequantize.op_type != "DequantizeLinear"
        or len(quantize.input) != 3
        or len(dequantize.input) != 3
        or len(quantize.output) != 1
        or len(dequantize.output) != 1
        or dequantize.input[0] != quantize.output[0]
        or consumers[quantize.output[0]] != 1
        or quantize.output[0] in graph_outputs
        or quantize.input[1:] != dequantize.input[1:]
        or any(name not in initializers for name in quantize.input[1:])
        or any(
            attribute.name == "block_size"
            for node in (quantize, dequantize)
            for attribute in node.attribute
        )
    ):
        return None

    scale = numpy_helper.to_array(initializers[quantize.input[1]])
    zero_point = numpy_helper.to_array(initializers[quantize.input[2]])
    if (
        scale.size != 1
        or zero_point.size != 1
        or not np.issubdtype(scale.dtype, np.floating)
        or not np.issubdtype(zero_point.dtype, np.integer)
        or not np.isfinite(scale).all()
        or not np.greater(scale, 0).all()
    ):
        return None
    return scale, zero_point


def _unique_name(base: str, used: set[str]) -> str:
    """Allocate a deterministic tensor name not already present in the graph."""
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _lower_qdq_pair(
    quantize: NodeProto,
    dequantize: NodeProto,
    scale: np.ndarray,
    zero_point: np.ndarray,
    initializers: dict[str, TensorProto],
    used_names: set[str],
) -> list[NodeProto]:
    """Lower one scalar Q-DQ pair to float-domain standard ONNX operators."""
    stem = dequantize.output[0]
    div_output = _unique_name(f"{stem}__quant_div", used_names)
    round_output = _unique_name(f"{stem}__quant_round", used_names)
    clip_output = _unique_name(f"{stem}__quant_clip", used_names)
    clip_min_name = _unique_name(f"{stem}__quant_clip_min", used_names)
    clip_max_name = _unique_name(f"{stem}__quant_clip_max", used_names)

    limits = np.iinfo(zero_point.dtype)
    zero = int(zero_point.reshape(-1)[0])
    initializers[clip_min_name] = numpy_helper.from_array(
        np.asarray(limits.min - zero, dtype=scale.dtype), name=clip_min_name
    )
    initializers[clip_max_name] = numpy_helper.from_array(
        np.asarray(limits.max - zero, dtype=scale.dtype), name=clip_max_name
    )

    return [
        helper.make_node(
            "Div",
            [quantize.input[0], quantize.input[1]],
            [div_output],
            name=f"{quantize.name or stem}__div",
        ),
        helper.make_node(
            "Round", [div_output], [round_output], name=f"{quantize.name or stem}__round"
        ),
        helper.make_node(
            "Clip",
            [round_output, clip_min_name, clip_max_name],
            [clip_output],
            name=f"{quantize.name or stem}__clip",
        ),
        helper.make_node(
            "Mul",
            [clip_output, quantize.input[1]],
            [dequantize.output[0]],
            name=f"{dequantize.name or stem}__mul",
        ),
    ]


def _canonicalize_quantization(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    graph_outputs: set[str],
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Canonicalize only quantization forms whose ONNX semantics are preserved.

    Fully static ``DequantizeLinear`` nodes become initializers.  Dynamic
    scalar ``QuantizeLinear -> DequantizeLinear`` pairs become
    ``Div -> Round -> Clip -> Mul`` in the floating-point domain.  Unsupported
    forms remain untouched so downstream consumers fail closed.
    """
    producers = {output: node for node in nodes for output in node.output}
    consumers: Counter[str] = Counter(name for node in nodes for name in node.input if name)
    eligible: dict[int, tuple[NodeProto, np.ndarray, np.ndarray]] = {}
    removed_quantize_ids: set[int] = set()

    for node in nodes:
        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        producer = producers.get(node.input[0])
        if producer is None:
            continue
        parameters = _eligible_qdq_pair(producer, node, initializers, consumers, graph_outputs)
        if parameters is not None:
            eligible[id(node)] = (producer, *parameters)
            removed_quantize_ids.add(id(producer))

    used_names = {name for node in nodes for name in (*node.input, *node.output) if name} | set(
        initializers
    )
    rewritten: list[NodeProto] = []
    for node in nodes:
        if id(node) in removed_quantize_ids:
            continue
        pair = eligible.get(id(node))
        if pair is not None:
            producer, scale, zero_point = pair
            rewritten.extend(
                _lower_qdq_pair(
                    producer,
                    node,
                    scale,
                    zero_point,
                    initializers,
                    used_names,
                )
            )
            continue
        folded = _fold_static_dequantize(node, initializers)
        if folded is not None:
            initializers[folded.name] = folded
            continue
        rewritten.append(node)

    return rewritten, initializers
