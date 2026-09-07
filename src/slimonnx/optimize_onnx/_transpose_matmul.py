"""Fuse rank-2 Transpose-MatMul-Transpose patterns."""

__docformat__ = "restructuredtext"
__all__ = ["_fuse_transpose_matmul_transpose"]

from collections import Counter

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, numpy_helper


def _get_perm(node: NodeProto) -> tuple[int, ...] | None:
    """Return a Transpose permutation when it is explicitly declared."""
    for attr in node.attribute:
        if attr.name == "perm":
            return tuple(attr.ints)
    return None


def _make_unique_name(base: str, used_names: set[str]) -> str:
    """Return a tensor name not already used by the graph."""
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _has_static_rank_two(shape: list[int] | None) -> bool:
    """Return whether a shape has two statically known dimensions."""
    return bool(
        shape is not None
        and len(shape) == 2
        and all(isinstance(dim, int) and dim >= 0 for dim in shape)
    )


def _is_safe_pattern(
    transpose_in: NodeProto,
    matmul: NodeProto,
    transpose_out: NodeProto,
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
    consumer_counts: Counter[str],
) -> bool:
    """Return whether a candidate has the exact proven rank-2 semantics."""
    if _get_perm(transpose_in) != (1, 0) or _get_perm(transpose_out) != (1, 0):
        return False
    if len(matmul.input) != 2 or matmul.input[1] != transpose_in.output[0]:
        return False
    if matmul.input[0] not in initializers:
        return False
    if len(initializers[matmul.input[0]].dims) != 2:
        return False
    tensor_names = (
        transpose_in.input[0],
        transpose_in.output[0],
        matmul.output[0],
        transpose_out.output[0],
    )
    if not all(_has_static_rank_two(data_shapes.get(name)) for name in tensor_names):
        return False
    return bool(
        consumer_counts[transpose_in.output[0]] == 1 and consumer_counts[matmul.output[0]] == 1
    )


def _fuse_transpose_matmul_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Replace ``T(W @ T(x))`` with ``x @ T(W)`` for safe rank-2 graphs.

    The two internal values must each have one consumer. The replacement keeps
    the final Transpose node name and output tensor, so downstream topology and
    graph outputs retain their public identity.

    :param nodes: Topologically ordered graph nodes.

    :param initializers: Initializer map, mutated with transposed weights.

    :param data_shapes: Static tensor shapes inferred for the current graph.

    :return: Rewritten nodes and initializer map.
    """
    producer = {output: node for node in nodes for output in node.output}
    consumer_counts = Counter(input_name for node in nodes for input_name in node.input)
    used_names = {name for node in nodes for name in (*node.input, *node.output)} | set(
        initializers
    )
    replacements: dict[int, NodeProto] = {}
    removed_ids: set[int] = set()

    for transpose_out in nodes:
        if transpose_out.op_type != "Transpose" or not transpose_out.input:
            continue
        matmul = producer.get(transpose_out.input[0])
        if matmul is None or matmul.op_type != "MatMul" or len(matmul.input) != 2:
            continue
        transpose_in = producer.get(matmul.input[1])
        if transpose_in is None or transpose_in.op_type != "Transpose":
            continue
        candidate_ids = {id(transpose_in), id(matmul), id(transpose_out)}
        if candidate_ids & (removed_ids | replacements.keys()):
            continue
        if not _is_safe_pattern(
            transpose_in,
            matmul,
            transpose_out,
            initializers,
            data_shapes,
            consumer_counts,
        ):
            continue

        weight_name = matmul.input[0]
        weight = numpy_helper.to_array(initializers[weight_name])
        transposed_name = _make_unique_name(f"{weight_name}_T", used_names)
        initializers[transposed_name] = numpy_helper.from_array(
            np.ascontiguousarray(weight.T),
            name=transposed_name,
        )

        replacement = onnx.NodeProto()
        replacement.CopyFrom(transpose_out)
        replacement.op_type = "MatMul"
        replacement.ClearField("attribute")
        replacement.ClearField("input")
        replacement.input.extend([transpose_in.input[0], transposed_name])

        replacements[id(transpose_out)] = replacement
        removed_ids.update((id(transpose_in), id(matmul)))
        if consumer_counts[weight_name] == 1:
            del initializers[weight_name]

    fused_nodes = []
    for node in nodes:
        if id(node) in removed_ids:
            continue
        fused_nodes.append(replacements.get(id(node), node))
    return fused_nodes, initializers
