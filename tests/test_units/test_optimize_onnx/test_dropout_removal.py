"""Tests for dropout removal optimization."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from slimonnx.optimize_onnx._dropout import (
    _build_dropout_mapping,
    _resolve_dropout_source,
    _update_node_inputs,
    remove_dropout,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (
    create_minimal_onnx_model,
    create_tensor_value_info,
    run_onnx_model,
)


class TestBuildDropoutMapping:
    """Test _build_dropout_mapping function."""

    def test_creates_mapping_with_single_node(self):
        """Test dropout mapping with single dropout node."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([dropout], [X], [Y])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        assert isinstance(mapping, dict)
        assert isinstance(to_remove, list)
        assert "Y" in mapping
        assert mapping["Y"] == "X"
        assert len(to_remove) == 1

    def test_creates_mapping_with_multiple_nodes(self):
        """Test dropout mapping with multiple dropout nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([dropout1, dropout2], [X], [Z])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        assert len(mapping) == 2
        assert len(to_remove) == 2
        assert "Y" in mapping
        assert "Z" in mapping

    def test_creates_empty_mapping_without_dropout(self):
        """Test dropout mapping with no dropout nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        assert len(mapping) == 0
        assert len(to_remove) == 0

    def test_creates_mapping_with_mixed_ops(self):
        """Test dropout mapping with mixed operations."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([dropout, relu], [X], [Z])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        assert len(mapping) == 1
        assert len(to_remove) == 1
        assert "Y" in mapping

    def test_keeps_static_training_dropout(self):
        """A scalar true training_mode forbids inference bypass."""
        training = numpy_helper.from_array(np.ones((), dtype=np.bool_), "training")
        dropout = helper.make_node(
            "Dropout",
            inputs=["X", "", "training"],
            outputs=["Y"],
        )

        mapping, to_remove = _build_dropout_mapping(
            [dropout],
            {"training": training},
        )

        assert mapping == {}
        assert to_remove == []

    def test_removes_static_inference_dropout(self):
        """A scalar false training_mode proves inference semantics."""
        training = numpy_helper.from_array(np.zeros((), dtype=np.bool_), "training")
        dropout = helper.make_node(
            "Dropout",
            inputs=["X", "", "training"],
            outputs=["Y"],
        )

        mapping, to_remove = _build_dropout_mapping(
            [dropout],
            {"training": training},
        )

        assert mapping == {"Y": "X"}
        assert to_remove == [dropout]

    def test_keeps_dynamic_training_mode(self):
        """A non-initializer training_mode cannot be assumed false."""
        dropout = helper.make_node(
            "Dropout",
            inputs=["X", "", "training"],
            outputs=["Y"],
        )

        mapping, to_remove = _build_dropout_mapping([dropout])

        assert mapping == {}
        assert to_remove == []

    def test_keeps_observable_mask(self):
        """A consumed mask output prevents removal."""
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y", "mask"])
        cast = helper.make_node(
            "Cast",
            inputs=["mask"],
            outputs=["mask_float"],
            to=TensorProto.FLOAT,
        )

        mapping, to_remove = _build_dropout_mapping([dropout, cast])

        assert mapping == {}
        assert to_remove == []

    def test_keeps_legacy_dropout_without_is_test_proof(self):
        """Legacy schema defaults to training unless is_test is proved."""
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])

        mapping, to_remove = _build_dropout_mapping(
            [dropout],
            default_inference_mode=False,
        )

        assert mapping == {}
        assert to_remove == []

    def test_removes_legacy_dropout_with_is_test(self):
        """Legacy is_test=1 is an explicit inference-mode proof."""
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"], is_test=1)

        mapping, to_remove = _build_dropout_mapping(
            [dropout],
            default_inference_mode=False,
        )

        assert mapping == {"Y": "X"}
        assert to_remove == [dropout]

    def test_keeps_custom_domain_dropout(self):
        """An identically named custom operator has no ONNX Dropout contract."""
        dropout = helper.make_node(
            "Dropout",
            inputs=["X"],
            outputs=["Y"],
            domain="example.custom",
        )

        mapping, to_remove = _build_dropout_mapping([dropout])

        assert mapping == {}
        assert to_remove == []


class TestUpdateNodeInputs:
    """Test _update_node_inputs function."""

    def test_updates_inputs_to_skip_dropout(self):
        """Test updating node inputs to bypass dropout."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([dropout, relu], [X], [Z])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        result = _update_node_inputs(nodes, to_remove, mapping)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Relu"
        assert result[0].input[0] == "X"

    def test_preserves_nodes_without_dropout(self):
        """Test updating nodes when no dropout present."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        result = _update_node_inputs(nodes, to_remove, mapping)

        assert len(result) == 1
        assert result[0].op_type == "Relu"

    def test_updates_all_consumers_of_dropout(self):
        """Test updating with multiple consumers of dropout."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
        W = create_tensor_value_info("W", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        relu1 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        relu2 = helper.make_node("Relu", inputs=["Y"], outputs=["W"])

        model = create_minimal_onnx_model([dropout, relu1, relu2], [X], [Z, W], [])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        result = _update_node_inputs(nodes, to_remove, mapping)

        assert len(result) == 2
        assert all(node.op_type == "Relu" for node in result)
        assert result[0].input[0] == "X"
        assert result[1].input[0] == "X"

    def test_updates_consumer_through_complete_chain(self):
        """A consumer of the second Dropout is rewired to the root source."""
        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["R"])
        nodes = [dropout1, dropout2, relu]

        mapping, to_remove = _build_dropout_mapping(nodes)
        result = _update_node_inputs(nodes, to_remove, mapping)

        assert len(result) == 1
        assert list(result[0].input) == ["X"]


class TestDropoutMappingResolution:
    """Test transitive Dropout rewiring."""

    def test_resolves_complete_dropout_chain(self):
        """Every bypassed output resolves to the original data source."""
        mapping = {"Y": "X", "Z": "Y"}

        assert _resolve_dropout_source("Z", mapping) == "X"


class TestRemoveDropout:
    """Test remove_dropout function."""

    def test_preserves_dropout_that_defines_graph_output(self):
        """A public primary output prevents identity-changing removal."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([dropout], [X], [Y])
        values = np.arange(3 * 32 * 32, dtype=np.float32).reshape(1, 3, 32, 32)

        result = remove_dropout(model)
        output = run_onnx_model(result, {"X": values})[0]

        assert isinstance(result, type(model))
        assert [node.op_type for node in result.graph.node] == ["Dropout"]
        assert result.graph.output[0].name == "Y"
        assert np.array_equal(output, values)

    def test_removes_dropout_from_chain(self):
        """Test dropout removal in dropout->relu chain."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([dropout, relu], [X], [Z])

        result = remove_dropout(model)
        assert all(node.op_type != "Dropout" for node in result.graph.node)
        # Relu should now take X directly
        relu_nodes = [n for n in result.graph.node if n.op_type == "Relu"]
        if relu_nodes:
            assert relu_nodes[0].input[0] == "X"

    def test_removes_multiple_dropout_nodes(self):
        """Test removal of a complete Dropout chain before a consumer."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        R = create_tensor_value_info("R", "float32", [1, 3, 32, 32])

        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["R"])

        model = create_minimal_onnx_model([dropout1, dropout2, relu], [X], [R])

        result = remove_dropout(model)
        assert all(node.op_type != "Dropout" for node in result.graph.node)
        assert list(result.graph.node[0].input) == ["X"]

    def test_preserves_graph_without_dropout(self):
        """Test remove_dropout when no dropout present."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        orig_nodes = len(model.graph.node)

        result = remove_dropout(model)
        assert len(result.graph.node) == orig_nodes
        assert result.graph.node[0].op_type == "Relu"

    def test_inference_dropout_is_bypassed_end_to_end(self):
        """An inference Dropout is an exact no-op before a real consumer."""
        X = create_tensor_value_info("X", "float32", [4])
        R = create_tensor_value_info("R", "float32", [4])
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["R"])
        model = create_minimal_onnx_model([dropout, relu], [X], [R])
        values = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

        result = remove_dropout(model)
        output = run_onnx_model(result, {"X": values})[0]

        assert [node.op_type for node in result.graph.node] == ["Relu"]
        assert np.array_equal(output, np.maximum(values, 0.0))

    def test_training_mode_true_is_preserved_end_to_end(self):
        """Training Dropout remains even when ratio zero makes the run deterministic."""
        X = create_tensor_value_info("X", "float32", [4])
        Y = create_tensor_value_info("Y", "float32", [4])
        ratio = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "ratio")
        training = numpy_helper.from_array(np.ones((), dtype=np.bool_), "training")
        dropout = helper.make_node(
            "Dropout",
            inputs=["X", "ratio", "training"],
            outputs=["Y"],
        )
        model = create_minimal_onnx_model(
            [dropout],
            [X],
            [Y],
            [ratio, training],
        )
        values = np.arange(4, dtype=np.float32)

        result = remove_dropout(model)
        output = run_onnx_model(result, {"X": values})[0]

        assert [node.op_type for node in result.graph.node] == ["Dropout"]
        assert np.array_equal(output, values)

    def test_consumed_mask_prevents_removal_end_to_end(self):
        """An inference mask consumed by another node remains defined."""
        X = create_tensor_value_info("X", "float32", [4])
        Y = create_tensor_value_info("Y", "float32", [4])
        mask_float = create_tensor_value_info("mask_float", "float32", [4])
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y", "mask"])
        cast = helper.make_node(
            "Cast",
            inputs=["mask"],
            outputs=["mask_float"],
            to=TensorProto.FLOAT,
        )
        model = create_minimal_onnx_model([dropout, cast], [X], [Y, mask_float])
        values = np.arange(4, dtype=np.float32)

        result = remove_dropout(model)
        output, mask = run_onnx_model(result, {"X": values})

        assert [node.op_type for node in result.graph.node] == ["Dropout", "Cast"]
        assert np.array_equal(output, values)
        assert np.array_equal(mask, np.ones(4, dtype=np.float32))

    def test_graph_output_mask_prevents_removal_end_to_end(self):
        """A public mask output is observable even without an internal consumer."""
        X = create_tensor_value_info("X", "float32", [4])
        Y = create_tensor_value_info("Y", "float32", [4])
        mask = helper.make_tensor_value_info("mask", TensorProto.BOOL, [4])
        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y", "mask"])
        model = create_minimal_onnx_model([dropout], [X], [Y, mask])
        values = np.arange(4, dtype=np.float32)

        result = remove_dropout(model)
        output, output_mask = run_onnx_model(result, {"X": values})

        assert [node.op_type for node in result.graph.node] == ["Dropout"]
        assert np.array_equal(output, values)
        assert np.array_equal(output_mask, np.ones(4, dtype=np.bool_))

    def test_two_node_chain_preserves_graph_output_identity_end_to_end(self):
        """A removed chain resolves to the root while retaining output name."""
        X = create_tensor_value_info("X", "float32", [4])
        R = create_tensor_value_info("public_R", "float32", [4])
        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["public_R"])
        model = create_minimal_onnx_model([dropout1, dropout2, relu], [X], [R])
        values = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        result = remove_dropout(model)
        output = run_onnx_model(result, {"X": values})[0]

        assert result.graph.output[0].name == "public_R"
        assert [node.op_type for node in result.graph.node] == ["Relu"]
        assert list(result.graph.node[0].input) == ["X"]
        assert list(result.graph.node[0].output) == ["public_R"]
        assert np.array_equal(output, np.maximum(values, 0.0))

    def test_dropout_chain_with_public_tail_remains_connected(self):
        """A public tail Dropout stays while its removable predecessor is bypassed."""
        X = create_tensor_value_info("X", "float32", [4])
        Z = create_tensor_value_info("Z", "float32", [4])
        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])
        model = create_minimal_onnx_model([dropout1, dropout2], [X], [Z])
        values = np.arange(4, dtype=np.float32)

        result = remove_dropout(model)
        output = run_onnx_model(result, {"X": values})[0]

        assert result.graph.output[0].name == "Z"
        assert [node.op_type for node in result.graph.node] == ["Dropout"]
        assert list(result.graph.node[0].input) == ["X"]
        assert list(result.graph.node[0].output) == ["Z"]
        assert np.array_equal(output, values)
