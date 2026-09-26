"""Requested tensor identity survives fusion, reordering and serialization."""

__docformat__ = "restructuredtext"

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from slimonnx import OptimizationConfig, SlimONNX, get_value_map


def _make_model():
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])]
    outputs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2, 2])]
    arrays = {"w": [[[[2.0]]]], "scale": [1.5], "shift": [-0.25], "mean": [0.5], "var": [1.0]}
    initializers = [
        numpy_helper.from_array(np.asarray(v, dtype=np.float32), k) for k, v in arrays.items()
    ]
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["conv"], name="same", kernel_shape=[1, 1]),
        helper.make_node(
            "BatchNormalization", ["conv", "scale", "shift", "mean", "var"], ["bn"], name="same"
        ),
        helper.make_node("Relu", ["bn"], ["positive"], name="same"),
        helper.make_node("Neg", ["x"], ["negative"], name="same"),
        helper.make_node("Relu", ["negative"], ["other"], name="same"),
        helper.make_node("Add", ["positive", "other"], ["out"], name="same"),
    ]
    return helper.make_model(
        helper.make_graph(nodes, "branches", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 21)],
    )


@pytest.mark.parametrize("rename_twice", [False, True])
def test_value_map_survives_fusion_reordering_and_serialization(rename_twice):
    """Equal-shaped sibling activations must never be paired by position."""
    original = _make_model()
    payload = original.SerializeToString()
    config = OptimizationConfig(fuse_conv_bn=True, simplify_node_name=rename_twice)
    slim = SlimONNX().slim_model(original, config, trace_values=("conv", "bn", "positive", "other"))
    slim = onnx.load_from_string(slim.SerializeToString())
    mapping = get_value_map(slim)
    assert set(mapping) == {"conv", "bn", "positive", "other"}
    assert mapping["conv"] is None
    assert mapping["positive"] != mapping["other"]
    relus = [node.output[0] for node in slim.graph.node if node.op_type == "Relu"]
    assert relus == [mapping["other"], mapping["positive"]]
    assert not any(node.op_type == "BatchNormalization" for node in slim.graph.node)
    assert (
        next(node.output[0] for node in slim.graph.node if node.op_type == "Conv") == mapping["bn"]
    )
    onnx.checker.check_model(slim)
    original = onnx.load_from_string(payload)
    x = np.asarray([[[[-1.0, 0.0], [1.0, 2.0]]]], dtype=np.float32)
    surviving = {source: target for source, target in mapping.items() if target is not None}
    expected = ReferenceEvaluator(original).run(list(surviving), {"x": x})
    actual = ReferenceEvaluator(slim).run(list(surviving.values()), {slim.graph.input[0].name: x})
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
    # Tracking must not change even one operator or initializer byte.
    plain = SlimONNX().slim_model(onnx.load_from_string(payload), config)
    assert slim.graph.SerializeToString() == plain.graph.SerializeToString()
    assert get_value_map(plain) == {}


@pytest.mark.parametrize(
    ("requested", "match"),
    [(("missing",), "one source producer"), (("positive", "positive"), "duplicate")],
)
def test_trace_rejects_invalid_sources_before_mutation(requested, match):
    model = _make_model()
    before = model.SerializeToString()
    with pytest.raises(ValueError, match=match):
        SlimONNX().slim_model(model, trace_values=requested)
    assert model.SerializeToString() == before


def test_trace_rejects_ambiguous_producer():
    model = _make_model()
    model.graph.node.append(helper.make_node("Relu", ["x"], ["positive"]))
    with pytest.raises(ValueError, match="one source producer"):
        SlimONNX().slim_model(model, trace_values=("positive",))


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"x": "missing"}',
        '{"x":"out","x":null}',
        '{"x": "out", "y": "out"}',
        '{"x": 3}',
        '{"x": {"nested": "out"}}',
    ],
)
def test_reader_rejects_corrupt_mapping(payload):
    model = _make_model()
    entry = model.metadata_props.add()
    entry.key = "slimonnx.value_map.v1"
    entry.value = payload
    with pytest.raises(ValueError, match="value-map"):
        get_value_map(model)


def test_reader_rejects_duplicate_metadata():
    model = _make_model()
    for _ in range(2):
        entry = model.metadata_props.add()
        entry.key = "slimonnx.value_map.v1"
        entry.value = '{"source":"out"}'
    with pytest.raises(ValueError, match="ambiguous value-map metadata"):
        get_value_map(model)


@pytest.mark.parametrize("gemm_first", [False, True])
def test_bn_reshape_fusion_invalidates_repurposed_value(gemm_first):
    """A surviving spelling is not proof that the old value survived."""
    arrays = {
        "w": np.eye(2, dtype=np.float32),
        "b": np.zeros(2, dtype=np.float32),
        "scale": np.full(2, 2, dtype=np.float32),
        "shift": np.ones(2, dtype=np.float32),
        "mean": np.zeros(2, dtype=np.float32),
        "var": np.ones(2, dtype=np.float32),
        "shape": np.array([1, 2, 1, 1] if gemm_first else [1, 2], dtype=np.int64),
    }
    if gemm_first:
        nodes = [
            helper.make_node("Gemm", ["x", "w", "b"], ["middle"]),
            helper.make_node("Reshape", ["middle", "shape"], ["reshaped"]),
            helper.make_node(
                "BatchNormalization", ["reshaped", "scale", "shift", "mean", "var"], ["out"]
            ),
        ]
    else:
        nodes = [
            helper.make_node("BatchNormalization", ["x", "scale", "shift", "mean", "var"], ["bn"]),
            helper.make_node("Reshape", ["bn", "shape"], ["middle"]),
            helper.make_node("Gemm", ["middle", "w", "b"], ["out"]),
        ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "fusion",
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [1, 2] if gemm_first else [1, 2, 1, 1]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "out", TensorProto.FLOAT, [1, 2, 1, 1] if gemm_first else [1, 2]
                )
            ],
            [numpy_helper.from_array(v, k) for k, v in arrays.items()],
        ),
        opset_imports=[helper.make_opsetid("", 21)],
    )
    config = OptimizationConfig(
        fuse_gemm_reshape_bn=gemm_first, fuse_bn_reshape_gemm=not gemm_first
    )
    slim = SlimONNX().slim_model(model, config, trace_values=("middle", "out"))
    assert len(slim.graph.node) == 2
    assert get_value_map(slim) == {"middle": None, "out": slim.graph.output[0].name}


def test_trace_rejects_shape_changing_policy_before_mutation():
    model = _make_model()
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="shape-changing"):
        SlimONNX().slim_model(
            model,
            OptimizationConfig(simplify_conv_to_flatten_gemm=True),
            trace_values=("positive",),
        )
    assert model.SerializeToString() == before
