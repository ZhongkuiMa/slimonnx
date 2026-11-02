"""Tests for exact complementary-ReLU canonicalization."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from slimonnx import OptimizationConfig, SlimONNX


def _model(
    *,
    negative_weight_delta: float = 0.0,
    second_scale: float = 1.0,
    fork_first_affine: bool = False,
) -> onnx.ModelProto:
    base_weight = np.array(
        [
            [1.0, 2.0, -1.0],
            [-2.0, 0.5, 3.0],
        ],
        dtype=np.float32,
    )
    base_bias = np.array([0.25, -0.75], dtype=np.float32)
    first_weight = np.stack(
        (
            base_weight[1],
            -base_weight[1],
            base_weight[0],
            -base_weight[0],
        )
    )
    first_weight[3, 0] += negative_weight_delta
    first_bias = np.array(
        [base_bias[1], -base_bias[1], base_bias[0], -base_bias[0]],
        dtype=np.float32,
    )
    second_weight = second_scale * np.array(
        [
            [0.0, 0.0, 1.0, -1.0],
            [1.0, -1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    trailing_bias = np.array([0.1, -0.2], dtype=np.float32)

    nodes = [
        helper.make_node(
            "Gemm",
            ["input", "first_weight", "first_bias"],
            ["first_affine"],
            name="first",
            transB=1,
        ),
        helper.make_node("Relu", ["first_affine"], ["first_relu"], name="first_relu"),
        helper.make_node(
            "Gemm",
            ["first_relu", "second_weight", "trailing_bias"],
            ["second_affine"],
            name="second",
            transB=1,
        ),
        helper.make_node("Relu", ["second_affine"], ["output"], name="second_relu"),
    ]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])]
    if fork_first_affine:
        nodes.append(helper.make_node("Identity", ["first_affine"], ["fork"], name="fork"))
        outputs.append(helper.make_tensor_value_info("fork", TensorProto.FLOAT, [1, 4]))
    graph = helper.make_graph(
        nodes,
        "complementary_relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        outputs,
        [
            numpy_helper.from_array(first_weight, "first_weight"),
            numpy_helper.from_array(first_bias, "first_bias"),
            numpy_helper.from_array(second_weight, "second_weight"),
            numpy_helper.from_array(trailing_bias, "trailing_bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _slim(model: onnx.ModelProto) -> onnx.ModelProto:
    return SlimONNX().slim_model(
        copy.deepcopy(model),
        OptimizationConfig(collapse_complementary_relu=True),
    )


def test_collapses_exact_unit_complement_without_rounding_bias() -> None:
    original = _model()
    optimized = _slim(original)

    assert [node.op_type for node in optimized.graph.node] == ["Gemm", "Add", "Relu"]
    initializers = {
        value.name: numpy_helper.to_array(value) for value in optimized.graph.initializer
    }
    gemm, add, _ = optimized.graph.node
    assert initializers[gemm.input[1]].shape == (3, 2)
    np.testing.assert_array_equal(
        initializers[gemm.input[2]],
        np.array([0.25, -0.75], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        initializers[add.input[1]],
        np.array([0.1, -0.2], dtype=np.float32),
    )
    assert "second_weight_1" not in initializers
    onnx.checker.check_model(optimized)

    inputs = (
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, -2.0, 0.5]], dtype=np.float32),
        np.array([[-0.25, 0.75, -1.5]], dtype=np.float32),
    )
    original_eval = ReferenceEvaluator(original)
    optimized_eval = ReferenceEvaluator(optimized)
    optimized_input = optimized.graph.input[0].name
    for value in inputs:
        expected = original_eval.run(None, {"input": value})[0]
        actual = optimized_eval.run(None, {optimized_input: value})[0]
        np.testing.assert_array_equal(actual, expected)


def test_rejects_tolerance_only_relation() -> None:
    optimized = _slim(_model(negative_weight_delta=1e-6))

    assert [node.op_type for node in optimized.graph.node] == [
        "Gemm",
        "Relu",
        "Gemm",
        "Relu",
    ]


def test_rejects_non_unit_coefficients() -> None:
    optimized = _slim(_model(second_scale=0.5))

    assert [node.op_type for node in optimized.graph.node] == [
        "Gemm",
        "Relu",
        "Gemm",
        "Relu",
    ]


def test_rejects_forked_intermediate_value() -> None:
    optimized = _slim(_model(fork_first_affine=True))

    assert Counter(node.op_type for node in optimized.graph.node) == {
        "Gemm": 2,
        "Relu": 2,
        "Identity": 1,
    }
