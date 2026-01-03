"""Tests for reshape optimization operations."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._reshape import _resolve_reshape_negative_one

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestResolveReshapeNegativeOne:
    """Test _resolve_reshape_negative_one function."""

    def test_resolve_reshape_negative_one_single_dimension(self):
        """Test resolving -1 in reshape to single dimension."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4])

        shape = create_initializer("shape", np.array([-1, 3, 4], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1
        # Check that initializers were modified
        assert "shape" in initializers

    def test_resolve_reshape_negative_one_multiple_dims(self):
        """Test resolving -1 in reshape with multiple dimensions."""
        X = create_tensor_value_info("X", "float32", [1, 24])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 4])

        shape = create_initializer("shape", np.array([2, -1, 4], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 24], "Y": [2, 3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_no_negative_one(self):
        """Test reshape without -1 (no change needed)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4])

        shape = create_initializer("shape", np.array([1, 3, 4], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1
        # Original shape should remain unchanged
        assert "shape" in initializers

    def test_resolve_reshape_non_reshape_node(self):
        """Test with non-Reshape nodes (should skip)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 12])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 12]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_missing_shape_input(self):
        """Test reshape without second input (malformed, should skip)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4])

        reshape = helper.make_node("Reshape", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_shape_not_initializer(self):
        """Test when shape is not an initializer (dynamic shape, should skip)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4])

        reshape = helper.make_node("Reshape", inputs=["X", "shape_dynamic"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_output_shape_not_known(self):
        """Test when output shape is not in data_shapes (should skip)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4])

        shape = create_initializer("shape", np.array([-1, 3, 4], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12]}  # Y is not in data_shapes

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_output_shape_dynamic(self):
        """Test when output shape is dynamic (has 0, should skip)."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Y = create_tensor_value_info("Y", "float32", [0, 3, 4])

        shape = create_initializer("shape", np.array([-1, 3, 4], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [0, 3, 4]}  # Dynamic shape

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_output_shape_scalar(self):
        """Test when output shape is scalar int (not list)."""
        X = create_tensor_value_info("X", "float32", [12])
        Y = create_tensor_value_info("Y", "float32", [12])

        shape = create_initializer("shape", np.array([-1], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [12], "Y": 12}  # Y as scalar

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    def test_resolve_reshape_multiple_reshape_nodes(self):
        """Test with multiple reshape nodes in sequence."""
        X = create_tensor_value_info("X", "float32", [1, 12])
        Z = create_tensor_value_info("Z", "float32", [3, 4])

        shape1 = create_initializer("shape1", np.array([1, -1, 4], dtype=np.int64))
        shape2 = create_initializer("shape2", np.array([-1, 4], dtype=np.int64))

        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["Y"])
        reshape2 = helper.make_node("Reshape", inputs=["Y", "shape2"], outputs=["Z"])

        model = create_minimal_onnx_model([reshape1, reshape2], [X], [Z], [shape1, shape2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 12], "Y": [1, 3, 4], "Z": [3, 4]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 2

    def test_resolve_reshape_large_batch(self):
        """Test reshape with larger tensor shapes."""
        X = create_tensor_value_info("X", "float32", [128, 768])
        Y = create_tensor_value_info("Y", "float32", [128, 12, 64])

        shape = create_initializer("shape", np.array([128, 12, -1], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [128, 768], "Y": [128, 12, 64]}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1
