"""Tests for exact arithmetic canonicalization."""

from __future__ import annotations

__docformat__ = "restructuredtext"

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from slimonnx.optimize_onnx import optimize_onnx


def _make_mul_model() -> onnx.ModelProto:
    """Build two self-products and one genuinely binary product."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["xx"], name="square_x"),
        helper.make_node("Mul", ["z", "z"], ["zz"], name="square_z"),
        helper.make_node("Mul", ["x", "z"], ["xz"], name="product"),
        helper.make_node("Add", ["xx", "zz"], ["sum_squares"]),
        helper.make_node("Add", ["sum_squares", "xz"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "self_mul", [x, z], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_self_mul_canonicalization_preserves_exact_outputs() -> None:
    """Self-products become shared-exponent Pow while distinct Mul stays binary."""
    original = _make_mul_model()
    optimized = optimize_onnx(original)

    onnx.checker.check_model(optimized)
    assert [node.op_type for node in optimized.graph.node] == ["Pow", "Pow", "Mul", "Add", "Add"]

    powers = [node for node in optimized.graph.node if node.op_type == "Pow"]
    products = [node for node in optimized.graph.node if node.op_type == "Mul"]
    assert len({node.input[1] for node in powers}) == 1
    assert products[0].input == ["x", "z"]

    exponent_name = powers[0].input[1]
    exponent = next(init for init in optimized.graph.initializer if init.name == exponent_name)
    np.testing.assert_array_equal(numpy_helper.to_array(exponent), np.asarray(2, dtype=np.int64))

    inputs = {
        "x": np.asarray([[-2.0, -0.5, 0.0], [0.25, 1.5, 3.0]], dtype=np.float32),
        "z": np.asarray([[1.0, -3.0, 2.0], [-0.75, 0.5, 4.0]], dtype=np.float32),
    }
    expected = ReferenceEvaluator(original).run(None, inputs)
    actual = ReferenceEvaluator(optimized).run(None, inputs)
    np.testing.assert_array_equal(actual[0], expected[0])


def test_self_mul_canonicalization_is_idempotent() -> None:
    """A second optimization creates neither more Pow nodes nor constants."""
    once = optimize_onnx(_make_mul_model())
    twice = optimize_onnx(once)

    assert [node.op_type for node in twice.graph.node].count("Pow") == 2
    assert [node.op_type for node in twice.graph.node].count("Mul") == 1
    assert len(twice.graph.initializer) == len(once.graph.initializer) == 1
