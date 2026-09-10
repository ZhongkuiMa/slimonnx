"""Exact canonicalization of arithmetic graph structure."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__: list[str] = []

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import _make_unique_name

_POW2_EXPONENT_NAME = "slimonnx_pow2_exp"


def _canonicalize_self_mul(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Rewrite exact ``Mul(x, x)`` nodes as unary square expressions.

    ONNX represents square through ``Pow`` with a constant exponent. Keeping
    self-multiplication as a generic binary operation hides operand identity
    from downstream consumers and forces them onto a weaker bilinear surface.
    One shared scalar exponent is materialized only when the graph contains at
    least one eligible node.

    Custom-domain operators, malformed arity, and genuinely distinct operands
    are preserved unchanged.

    :param nodes: Graph nodes in topological order.
    :param initializers: Initializer map, mutated with the shared exponent.
    :return: Nodes with every standard-domain self-Mul represented as Pow(2).
    """
    eligible = [
        node
        for node in nodes
        if node.op_type == "Mul"
        and not node.domain
        and len(node.input) == 2
        and bool(node.input[0])
        and node.input[0] == node.input[1]
    ]
    if not eligible:
        return nodes

    used_names = {name for node in nodes for name in (*node.input, *node.output)} | set(
        initializers
    )
    exponent_name = _make_unique_name(_POW2_EXPONENT_NAME, used_names)
    initializers[exponent_name] = onnx.numpy_helper.from_array(
        np.asarray(2, dtype=np.int64),
        exponent_name,
    )

    eligible_ids = {id(node) for node in eligible}
    rewritten: list[NodeProto] = []
    for node in nodes:
        if id(node) not in eligible_ids:
            rewritten.append(node)
            continue

        square = NodeProto()
        square.CopyFrom(node)
        square.op_type = "Pow"
        del square.input[:]
        square.input.extend((node.input[0], exponent_name))
        del square.attribute[:]
        rewritten.append(square)

    return rewritten
