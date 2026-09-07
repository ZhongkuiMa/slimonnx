"""Tests for Transpose-MatMul-Transpose fusion."""

__docformat__ = "restructuredtext"

import copy
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from _helpers import create_initializer, create_minimal_onnx_model, create_tensor_value_info
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx import optimize_onnx


def _make_pattern_model(
    *,
    dynamic_weight: bool = False,
    fanout: bool = False,
    input_perm: tuple[int, int] = (1, 0),
) -> onnx.ModelProto:
    """Build a valid rank-2 constant-left matrix multiplication pattern."""
    inputs = [create_tensor_value_info("X", "float32", [2, 3])]
    initializers = []
    if dynamic_weight:
        inputs.append(create_tensor_value_info("W", "float32", [4, 3]))
    else:
        initializers.append(create_initializer("W", np.arange(12, dtype=np.float32).reshape(4, 3)))

    transpose_in = helper.make_node(
        "Transpose",
        inputs=["X"],
        outputs=["X_T"],
        name="transpose_input",
        perm=input_perm,
    )
    matmul = helper.make_node(
        "MatMul",
        inputs=["W", "X_T"],
        outputs=["WX_T"],
        name="constant_left_matmul",
    )
    transpose_out = helper.make_node(
        "Transpose",
        inputs=["WX_T"],
        outputs=["Y"],
        name="transpose_output",
        perm=[1, 0],
    )
    nodes = [transpose_in, matmul, transpose_out]
    outputs = [create_tensor_value_info("Y", "float32", [2, 4])]
    if fanout:
        nodes.insert(
            1,
            helper.make_node("Relu", inputs=["X_T"], outputs=["side"], name="side_user"),
        )
        outputs.append(create_tensor_value_info("side", "float32", [3, 2]))
    return create_minimal_onnx_model(nodes, inputs, outputs, initializers)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Execute an ONNX model on the CPU."""
    session = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return cast("list[np.ndarray]", session.run(None, feeds))


def test_fuses_safe_pattern_and_preserves_interface():
    """Fuse the exact pattern while retaining the terminal identity."""
    original = _make_pattern_model()
    optimized = optimize_onnx(
        copy.deepcopy(original),
        fuse_transpose_matmul_transpose=True,
        has_batch_dim=False,
    )

    onnx.checker.check_model(optimized)
    assert [node.op_type for node in optimized.graph.node] == ["MatMul"]
    fused = optimized.graph.node[0]
    assert fused.name == "transpose_output"
    assert list(fused.output) == ["Y"]
    assert list(fused.input[:1]) == ["X"]
    transposed_weight = numpy_helper.to_array(optimized.graph.initializer[0])
    np.testing.assert_array_equal(
        transposed_weight,
        np.arange(12, dtype=np.float32).reshape(4, 3).T,
    )

    test_input = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual = _run(optimized, {"X": test_input})
    expected = _run(original, {"X": test_input})
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)


def test_does_not_fuse_without_flag():
    """Keep the pattern when its public optimization flag is disabled."""
    optimized = optimize_onnx(_make_pattern_model(), has_batch_dim=False)

    assert [node.op_type for node in optimized.graph.node] == [
        "Transpose",
        "MatMul",
        "Transpose",
    ]


@pytest.mark.parametrize(
    ("dynamic_weight", "fanout"),
    [
        pytest.param(True, False, id="dynamic_weight"),
        pytest.param(False, True, id="internal_fanout"),
    ],
)
def test_does_not_fuse_unproven_pattern(dynamic_weight, fanout):
    """Keep patterns without a static weight or single-use internal edge."""
    model = _make_pattern_model(dynamic_weight=dynamic_weight, fanout=fanout)
    optimized = optimize_onnx(
        model,
        fuse_transpose_matmul_transpose=True,
        has_batch_dim=False,
    )

    assert sum(node.op_type == "Transpose" for node in optimized.graph.node) == 2
    assert sum(node.op_type == "MatMul" for node in optimized.graph.node) == 1


def test_does_not_fuse_wrong_permutation():
    """Keep a valid graph whose input Transpose is not the swap permutation."""
    inputs = [create_tensor_value_info("X", "float32", [3, 3])]
    initializers = [create_initializer("W", np.ones((4, 3), dtype=np.float32))]
    nodes = [
        helper.make_node("Transpose", ["X"], ["X_T"], perm=[0, 1]),
        helper.make_node("MatMul", ["W", "X_T"], ["WX_T"]),
        helper.make_node("Transpose", ["WX_T"], ["Y"], perm=[1, 0]),
    ]
    outputs = [create_tensor_value_info("Y", "float32", [3, 4])]
    model = create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    optimized = optimize_onnx(
        model,
        fuse_transpose_matmul_transpose=True,
        has_batch_dim=False,
    )

    assert [node.op_type for node in optimized.graph.node] == [
        "Transpose",
        "MatMul",
        "Transpose",
    ]
