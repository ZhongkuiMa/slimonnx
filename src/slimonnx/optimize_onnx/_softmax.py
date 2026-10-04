"""Canonicalize exact legacy Softmax conversion scaffolds."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["_canonicalize_legacy_softmax"]

from collections import Counter

from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper


def _axis(node: NodeProto, default: int) -> int:
    """Return one integer axis attribute or its schema default."""
    return next((int(attr.i) for attr in node.attribute if attr.name == "axis"), default)


def _allowzero(node: NodeProto) -> int:
    """Return the Reshape ``allowzero`` attribute."""
    return next((int(attr.i) for attr in node.attribute if attr.name == "allowzero"), 0)


def _canonicalize_legacy_softmax(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
    output_nodes: list[ValueInfoProto],
) -> list[NodeProto]:
    """Replace an exact converter-owned legacy Softmax scaffold.

    ONNX version conversion represents an old Softmax on the final axis as
    ``Shape(X); Flatten(X); Softmax; Reshape(..., Shape(X))``. When the
    Flatten axis is the final input axis, the scaffold is exactly a modern
    last-axis Softmax. Only single-consumer, non-observable scaffolds are
    rewritten; every other graph is left unchanged.
    """
    producers = {name: node for node in nodes for name in node.output if name}
    consumer_count = Counter(name for node in nodes for name in node.input if name)
    graph_outputs = {output.name for output in output_nodes}
    replacements: dict[int, NodeProto] = {}
    removed: set[int] = set()

    for reshape in nodes:
        if reshape.op_type != "Reshape" or len(reshape.input) != 2 or len(reshape.output) != 1:
            continue
        softmax = producers.get(reshape.input[0])
        shape = producers.get(reshape.input[1])
        shape_initializer = initializers.get(reshape.input[1])
        if softmax is None or softmax.op_type != "Softmax":
            continue
        if len(softmax.input) != 1 or len(softmax.output) != 1:
            continue
        flatten = producers.get(softmax.input[0])
        if flatten is None or flatten.op_type != "Flatten":
            continue
        if len(flatten.input) != 1 or len(flatten.output) != 1:
            continue
        source = flatten.input[0]
        if shape is not None:
            if shape.op_type != "Shape" or len(shape.input) != 1 or len(shape.output) != 1:
                continue
            if shape.input[0] != source:
                continue
            shape_output = shape.output[0]
        elif shape_initializer is not None:
            shape_output = shape_initializer.name
        else:
            continue
        if any(
            consumer_count[name] != 1
            for name in (shape_output, flatten.output[0], softmax.output[0])
        ):
            continue
        if any(
            name in graph_outputs for name in (shape_output, flatten.output[0], softmax.output[0])
        ):
            continue
        if _allowzero(reshape) != 0:
            continue
        source_shape = data_shapes.get(source)
        if not isinstance(source_shape, list) or len(source_shape) < 2 or 0 in source_shape:
            continue
        if shape_initializer is not None and tuple(
            numpy_helper.to_array(shape_initializer)
        ) != tuple(source_shape):
            continue
        rank = len(source_shape)
        flatten_axis = _axis(flatten, 1)
        if flatten_axis < 0:
            flatten_axis += rank
        if flatten_axis != rank - 1:
            continue
        if _axis(softmax, -1) not in (-1, 1):
            continue

        canonical = NodeProto()
        canonical.CopyFrom(softmax)
        canonical.ClearField("input")
        canonical.input.append(source)
        canonical.ClearField("output")
        canonical.output.extend(reshape.output)
        canonical.ClearField("attribute")
        canonical.attribute.append(helper.make_attribute("axis", -1))
        replacements[id(flatten)] = canonical
        removed.update((id(flatten), id(softmax), id(reshape)))
        if shape is not None:
            removed.add(id(shape))
        else:
            del initializers[shape_output]

    return [
        replacements.get(id(node), node)
        for node in nodes
        if id(node) not in removed or id(node) in replacements
    ]
