"""Tests for conservative QuantizeLinear/DequantizeLinear canonicalization."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from slimonnx.optimize_onnx import optimize_onnx
from slimonnx.optimize_onnx._quantize import _canonicalize_quantization


def _run(model: onnx.ModelProto, values: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": values})[0]


def _model(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    *,
    input_shape: list[int],
    output_shape: list[int],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "quantization",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def test_scalar_qdq_lowers_to_exact_float_domain_ops() -> None:
    scale = numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(128, dtype=np.uint8), name="zero")
    quantize = helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx"], name="q")
    dequantize = helper.make_node("DequantizeLinear", ["qx", "scale", "zero"], ["y"], name="dq")
    original = _model(
        [quantize, dequantize], [scale, zero], input_shape=[1, 9], output_shape=[1, 9]
    )

    initializers = {item.name: item for item in original.graph.initializer}
    nodes, initializers = _canonicalize_quantization(list(original.graph.node), initializers, {"y"})
    lowered = _model(nodes, list(initializers.values()), input_shape=[1, 9], output_shape=[1, 9])
    onnx.checker.check_model(lowered)

    assert [node.op_type for node in nodes] == ["Div", "Round", "Clip", "Mul"]
    values = np.asarray(
        [[-100.0, -32.0, -0.375, -0.125, 0.125, 0.375, 31.75, 40.0, 100.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(_run(lowered, values), _run(original, values))


def test_public_optimizer_applies_quantization_canonicalization() -> None:
    scale = numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(128, dtype=np.uint8), name="zero")
    original = _model(
        [
            helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx"]),
            helper.make_node("DequantizeLinear", ["qx", "scale", "zero"], ["y"]),
        ],
        [scale, zero],
        input_shape=[1, 3],
        output_shape=[1, 3],
    )

    optimized = optimize_onnx(original, canonicalize_quantization=True)

    assert [node.op_type for node in optimized.graph.node] == ["Div", "Round", "Clip", "Mul"]
    values = np.asarray([[-0.125, 0.125, 100.0]], dtype=np.float32)
    np.testing.assert_array_equal(_run(optimized, values), _run(original, values))


def test_static_per_axis_dequantize_folds_exactly() -> None:
    data = numpy_helper.from_array(np.asarray([[0, 10], [20, 30]], dtype=np.uint8), name="data")
    scale = numpy_helper.from_array(np.asarray([0.5, 0.25], dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray([2, 4], dtype=np.uint8), name="zero")
    dequantize = helper.make_node("DequantizeLinear", ["data", "scale", "zero"], ["y"], axis=0)

    nodes, initializers = _canonicalize_quantization(
        [dequantize], {item.name: item for item in (data, scale, zero)}, {"y"}
    )

    assert nodes == []
    expected = (numpy_helper.to_array(data).astype(np.float32) - np.array([[2], [4]])) * np.array(
        [[0.5], [0.25]], dtype=np.float32
    )
    np.testing.assert_array_equal(numpy_helper.to_array(initializers["y"]), expected)


def test_static_singleton_parameters_fold_as_per_tensor() -> None:
    data = numpy_helper.from_array(np.asarray([24], dtype=np.int32), name="data")
    scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int32), name="zero")
    original = _model(
        [helper.make_node("DequantizeLinear", ["data", "scale", "zero"], ["y"])],
        [data, scale, zero],
        input_shape=[1],
        output_shape=[1],
    )

    optimized = optimize_onnx(original, canonicalize_quantization=True)

    assert list(optimized.graph.node) == []
    values = np.asarray([0.0], dtype=np.float32)
    np.testing.assert_array_equal(_run(optimized, values), _run(original, values))


def test_per_axis_dynamic_qdq_remains_untouched() -> None:
    scale = numpy_helper.from_array(np.asarray([0.25, 0.5], dtype=np.float32), name="scale")
    zero = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int8), name="zero")
    quantize = helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["qx"], axis=1, name="q")
    dequantize = helper.make_node(
        "DequantizeLinear", ["qx", "scale", "zero"], ["y"], axis=1, name="dq"
    )

    nodes, _ = _canonicalize_quantization(
        [quantize, dequantize], {item.name: item for item in (scale, zero)}, {"y"}
    )

    assert [node.op_type for node in nodes] == ["QuantizeLinear", "DequantizeLinear"]
