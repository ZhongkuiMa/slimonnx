"""Tests for dropout removal optimization."""

import sys
from pathlib import Path
from typing import Any

from onnx import helper

from slimonnx.optimize_onnx._dropout import (
    _build_dropout_mapping,
    _update_graph_outputs,
    _update_node_inputs,
    remove_dropout,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestBuildDropoutMapping:
    """Test _build_dropout_mapping function."""

    def test_build_dropout_mapping_single(self):
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

    def test_build_dropout_mapping_multiple(self):
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

    def test_build_dropout_mapping_no_dropout(self):
        """Test dropout mapping with no dropout nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)

        mapping, to_remove = _build_dropout_mapping(nodes)
        assert len(mapping) == 0
        assert len(to_remove) == 0

    def test_build_dropout_mapping_mixed(self):
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


class TestUpdateNodeInputs:
    """Test _update_node_inputs function."""

    def test_update_node_inputs_dropout_bypass(self):
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

    def test_update_node_inputs_no_dropout(self):
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

    def test_update_node_inputs_multiple_consumers(self):
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


class TestUpdateGraphOutputs:
    """Test _update_graph_outputs function."""

    def test_update_graph_outputs_dropout_output(self):
        """Test updating graph outputs that are dropout outputs."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([dropout], [X], [Y])
        outputs = list(model.graph.output)

        mapping = {"Y": "X"}
        result = _update_graph_outputs(outputs, mapping)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].name == "X"

    def test_update_graph_outputs_no_dropout(self):
        """Test updating graph outputs with no dropout."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        outputs = list(model.graph.output)

        mapping: dict[str, Any] = {}
        result = _update_graph_outputs(outputs, mapping)

        assert len(result) == 1
        assert result[0].name == "Y"


class TestRemoveDropout:
    """Test remove_dropout function."""

    def test_remove_dropout_basic(self):
        """Test basic dropout removal."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        dropout = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([dropout], [X], [Y])

        result = remove_dropout(model)
        assert isinstance(result, type(model))
        # Check that dropout is removed
        assert all(node.op_type != "Dropout" for node in result.graph.node)

    def test_remove_dropout_with_relu(self):
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

    def test_remove_dropout_multiple(self):
        """Test removal of multiple dropout nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        dropout1 = helper.make_node("Dropout", inputs=["X"], outputs=["Y"])
        dropout2 = helper.make_node("Dropout", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([dropout1, dropout2], [X], [Z])

        result = remove_dropout(model)
        assert all(node.op_type != "Dropout" for node in result.graph.node)

    def test_remove_dropout_no_dropout(self):
        """Test remove_dropout when no dropout present."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        orig_nodes = len(model.graph.node)

        result = remove_dropout(model)
        assert len(result.graph.node) == orig_nodes
        assert result.graph.node[0].op_type == "Relu"
