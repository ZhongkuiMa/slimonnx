"""Tests for reshape pattern detection."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.pattern_detect.reshape_chains import (
    detect_consecutive_reshape,
)
from slimonnx.pattern_detect.reshape_negative_one import (
    detect_reshape_with_negative_one,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDetectReshapeWithNegativeOne:
    """Test detect_reshape_with_negative_one function."""

    @pytest.mark.parametrize(
        ("shape_array", "expected_count", "case_id"),
        [
            pytest.param(
                np.array([2, 3, 2], dtype=np.int64),
                0,
                "no_negative_one",
            ),
            pytest.param(
                np.array([2, -1, 2], dtype=np.int64),
                1,
                "with_negative_one",
            ),
        ],
    )
    def test_detects_reshape_with_negative_one(self, shape_array, expected_count, case_id):
        """Test detection of reshape with -1 dimension."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 2])

        shape = create_initializer("shape", shape_array)

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [2, 6], "Y": [2, 3, 2]}

        result = detect_reshape_with_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result, list)
        assert len(result) == expected_count

    def test_no_detection_in_model_with_other_ops(self):
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
        assert isinstance(result, list)
        assert len(result) == 0


class TestDetectConsecutiveReshape:
    """Test detect_consecutive_reshape function."""

    @pytest.mark.parametrize(
        ("output_shape", "shape_array"),
        [
            pytest.param([2, 3, 2], np.array([2, 3, 2], dtype=np.int64), id="to_3d"),
            pytest.param([12], np.array([12], dtype=np.int64), id="to_1d"),
        ],
    )
    def test_no_chain_with_single_reshape(self, output_shape, shape_array):
        """Test detection with single reshape (not a chain)."""
        X = create_tensor_value_info("X", "float32", [2, 6])
        Y = create_tensor_value_info("Y", "float32", output_shape)

        shape = create_initializer("shape", shape_array)

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)

        result = detect_consecutive_reshape(nodes)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_identifies_chain_of_two_reshapes(self):
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
        assert isinstance(result, list)
        assert len(result) == 1

    def test_identifies_chain_of_three_reshapes(self):
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
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_chain_with_interleaved_ops(self):
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
        assert isinstance(result, list)
        assert len(result) == 0
