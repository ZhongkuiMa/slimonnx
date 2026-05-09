"""Tests for reshape optimization operations."""

import sys
from pathlib import Path

import numpy as np
import pytest
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

    @pytest.mark.parametrize(
        ("x_shape", "y_shape", "shape_vals", "node_factory", "check_shape_key"),
        [
            pytest.param(
                [1, 12],
                [1, 3, 4],
                np.array([-1, 3, 4], dtype=np.int64),
                "reshape",
                True,
                id="negative_one_single_dim",
            ),
            pytest.param(
                [1, 24],
                [2, 3, 4],
                np.array([2, -1, 4], dtype=np.int64),
                "reshape",
                False,
                id="negative_one_multiple_dims",
            ),
            pytest.param(
                [1, 12],
                [1, 3, 4],
                np.array([1, 3, 4], dtype=np.int64),
                "reshape",
                True,
                id="no_negative_one",
            ),
            pytest.param([1, 12], [1, 12], None, "relu", False, id="non_reshape_node"),
            pytest.param(
                [1, 12], [1, 3, 4], None, "reshape_no_shape", False, id="missing_shape_input"
            ),
        ],
    )
    def test_handles_various_reshape_configs(
        self, x_shape, y_shape, shape_vals, node_factory, check_shape_key
    ):
        """Verify _resolve_reshape_negative_one handles each variant of reshape input correctly."""
        X = create_tensor_value_info("X", "float32", x_shape)
        Y = create_tensor_value_info("Y", "float32", y_shape)

        inits = []
        if shape_vals is not None:
            inits.append(create_initializer("shape", shape_vals))

        if node_factory == "reshape":
            node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])
        elif node_factory == "reshape_no_shape":
            node = helper.make_node("Reshape", inputs=["X"], outputs=["Y"])
        else:
            node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([node], [X], [Y], inits)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": x_shape, "Y": y_shape}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)

        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1
        if check_shape_key:
            assert "shape" in initializers

    def test_skips_when_shape_not_initializer(self):
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

    @pytest.mark.parametrize(
        ("x_shape", "y_shape", "shape_array", "data_shapes"),
        [
            pytest.param(
                [1, 12],
                [1, 3, 4],
                np.array([-1, 3, 4], dtype=np.int64),
                {"X": [1, 12]},  # Y not in data_shapes
                id="output_shape_unknown",
            ),
            pytest.param(
                [1, 12],
                [0, 3, 4],
                np.array([-1, 3, 4], dtype=np.int64),
                {"X": [1, 12], "Y": [0, 3, 4]},  # Dynamic shape
                id="output_shape_dynamic",
            ),
            pytest.param(
                [12],
                [12],
                np.array([-1], dtype=np.int64),
                {"X": [12], "Y": 12},  # Scalar output shape
                id="scalar_output_shape",
            ),
            pytest.param(
                [128, 768],
                [128, 12, 64],
                np.array([128, 12, -1], dtype=np.int64),
                {"X": [128, 768], "Y": [128, 12, 64]},  # Large batch
                id="large_batch_tensors",
            ),
        ],
    )
    def test_handles_various_shape_scenarios(self, x_shape, y_shape, shape_array, data_shapes):
        """Verify _resolve_reshape_negative_one handles each shape scenario correctly."""
        X = create_tensor_value_info("X", "float32", x_shape)
        Y = create_tensor_value_info("Y", "float32", y_shape)

        shape = create_initializer("shape", shape_array)

        reshape = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape], [X], [Y], [shape])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result_nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
        assert isinstance(result_nodes, list)
        assert len(result_nodes) == 1

    # [REVIEW] Deleted: test_skips_when_output_shape_unknown (merged into test_handles_various_shape_scenarios)
    # [REVIEW] Deleted: test_skips_when_output_shape_dynamic (merged into test_handles_various_shape_scenarios)
    # [REVIEW] Deleted: test_handles_scalar_output_shape (merged into test_handles_various_shape_scenarios)
    # [REVIEW] Deleted: test_handles_large_batch_tensors (merged into test_handles_various_shape_scenarios)

    def test_processes_chained_reshape_nodes(self):
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

    # [REVIEW] Deleted: test_handles_large_batch_tensors (merged into test_handles_various_shape_scenarios)
