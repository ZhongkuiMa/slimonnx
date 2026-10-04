"""Tests for exact legacy Softmax scaffold canonicalization."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from slimonnx import OptimizationConfig, SlimONNX
from slimonnx.optimize_onnx._softmax import _canonicalize_legacy_softmax


def _model(*, flatten_axis: int = 3, softmax_axis: int = -1, extra_consumer: bool = False):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 5])
    shape = helper.make_node("Shape", ["x"], ["shape"])
    flatten = helper.make_node("Flatten", ["x"], ["flat"], axis=flatten_axis)
    softmax = helper.make_node("Softmax", ["flat"], ["prob"], axis=softmax_axis)
    reshape = helper.make_node("Reshape", ["prob", "shape"], ["y"])
    nodes = [shape, flatten, softmax, reshape]
    outputs = [y]
    if extra_consumer:
        tap = helper.make_tensor_value_info("tap", TensorProto.FLOAT, [12, 5])
        nodes.append(helper.make_node("Identity", ["flat"], ["tap"]))
        outputs.append(tap)
    graph = helper.make_graph(nodes, "legacy-softmax", [x], outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _run(model, value):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: value})


def test_exact_last_axis_scaffold_is_replaced_and_preserves_outputs():
    original = _model()
    optimized = SlimONNX().slim_model(original, OptimizationConfig())
    onnx.checker.check_model(optimized)
    assert [node.op_type for node in optimized.graph.node] == ["Softmax"]
    assert helper.get_attribute_value(optimized.graph.node[0].attribute[0]) == -1
    value = np.random.default_rng(0).standard_normal((1, 3, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(
        _run(original, value)[0], _run(optimized, value)[0], rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize(
    ("flatten_axis", "softmax_axis", "extra_consumer"),
    [(2, -1, False), (3, 0, False), (3, -1, True)],
)
def test_non_exact_or_observable_scaffolds_are_retained(flatten_axis, softmax_axis, extra_consumer):
    model = _model(
        flatten_axis=flatten_axis,
        softmax_axis=softmax_axis,
        extra_consumer=extra_consumer,
    )
    shapes: dict[str, int | list[int]] = {"x": [1, 3, 4, 5]}
    nodes = _canonicalize_legacy_softmax(
        list(model.graph.node), {}, shapes, list(model.graph.output)
    )
    assert [node.op_type for node in nodes[:4]] == ["Shape", "Flatten", "Softmax", "Reshape"]
