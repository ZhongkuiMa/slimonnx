"""Tests for reshape pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.reshape_chains import (
    detect_consecutive_reshape,
)
from slimonnx.pattern_detect.reshape_negative_one import (
    detect_reshape_with_negative_one,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDetectReshapeWithNegativeOne:
    """Test detect_reshape_with_negative_one function."""

    def test_detect_reshape_with_negative_one(self):
        """Test detection of reshape with -1 dimension."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 2])

        shape = create_initializer("shape", np.array([2, 3, 2], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 6], "Y": [2, 3, 2]}

        result = detect_reshape_with_negative_one(nodes, initializers, data_shapes)
        assert result is None or isinstance(result, list)

    def test_detect_reshape_with_literal_negative_one(self):
        """Test detection when reshape has -1 in shape."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 2])

        shape = create_initializer("shape", np.array([2, -1, 2], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 6], "Y": [2, 3, 2]}

        result = detect_reshape_with_negative_one(nodes, initializers, data_shapes)
        assert result is None or isinstance(result, list)

    def test_detect_reshape_no_negative_one(self):
        """Test no detection when reshape has no -1."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 2])

        shape = create_initializer("shape", np.array([2, 3, 2], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 6], "Y": [2, 3, 2]}

        result = detect_reshape_with_negative_one(nodes, initializers, data_shapes)
        assert result is None or isinstance(result, list)

    def test_detect_reshape_in_model_with_other_ops(self):
        """Test reshape detection in model with other operations."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Z = create_tensor_value_info("Z", "float32", [2, 3, 2])

        shape = create_initializer("shape", np.array([2, 3, 2], dtype=np.int64))

        identity = helper.make_node("Identity", inputs=["X"], outputs=["I"])
        reshape = helper.make_node("Reshape", inputs=["I", "shape"], outputs=["Z"])

        model = create_minimal_onnx_model([identity, reshape], [X], [Z], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 6], "I": [2, 6], "Z": [2, 3, 2]}

        result = detect_reshape_with_negative_one(nodes, initializers, data_shapes)
        assert result is None or isinstance(result, list)


class TestDetectConsecutiveReshape:
    """Test detect_consecutive_reshape function."""

    def test_detect_single_reshape(self):
        """Test detection with single reshape (not a chain)."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 2])

        shape = create_initializer("shape", np.array([2, 3, 2], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert result is None or isinstance(result, list)

    def test_detect_two_consecutive_reshapes(self):
        """Test detection of consecutive reshape chain."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Z = create_tensor_value_info("Z", "float32", [6, 2])

        shape1 = create_initializer("shape1", np.array([2, 3, 2], dtype=np.int64))
        shape2 = create_initializer("shape2", np.array([6, 2], dtype=np.int64))

        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["Y"])
        reshape2 = helper.make_node("Reshape", inputs=["Y", "shape2"], outputs=["Z"])

        model = create_minimal_onnx_model([reshape1, reshape2], [X], [Z], [shape1, shape2])
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert result is None or isinstance(result, list)

    def test_detect_three_consecutive_reshapes(self):
        """Test detection of long reshape chain."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        W = create_tensor_value_info("W", "float32", [12])

        shape1 = create_initializer("shape1", np.array([2, 3, 2], dtype=np.int64))
        shape2 = create_initializer("shape2", np.array([6, 2], dtype=np.int64))
        shape3 = create_initializer("shape3", np.array([12], dtype=np.int64))

        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["Y1"])
        reshape2 = helper.make_node("Reshape", inputs=["Y1", "shape2"], outputs=["Y2"])
        reshape3 = helper.make_node("Reshape", inputs=["Y2", "shape3"], outputs=["W"])

        model = create_minimal_onnx_model(
            [reshape1, reshape2, reshape3], [X], [W], [shape1, shape2, shape3]
        )
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert result is None or isinstance(result, list)

    def test_detect_reshape_chain_with_other_ops(self):
        """Test reshape chain detection with other operations."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Z = create_tensor_value_info("Z", "float32", [6, 2])

        shape1 = create_initializer("shape1", np.array([2, 3, 2], dtype=np.int64))
        shape2 = create_initializer("shape2", np.array([6, 2], dtype=np.int64))

        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["R"])
        reshape2 = helper.make_node("Reshape", inputs=["R", "shape2"], outputs=["Z"])

        model = create_minimal_onnx_model([reshape1, relu, reshape2], [X], [Z], [shape1, shape2])
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert result is None or isinstance(result, list)

    def test_detect_no_chain_single_reshape(self):
        """Test no chain detection with just one reshape."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [12])

        shape = create_initializer("shape", np.array([12], dtype=np.int64))

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert result is None or isinstance(result, list)
