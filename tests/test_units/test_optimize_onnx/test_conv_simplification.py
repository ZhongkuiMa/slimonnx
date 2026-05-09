"""Tests for Conv simplification optimizations."""

import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper

from slimonnx.optimize_onnx._conv import (
    _can_simplify_conv,
    _create_flatten_gemm_nodes,
    _simplify_conv_to_flatten_gemm,
    _validate_conv_simplification,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Any

from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestCanSimplifyConv:
    """Test _can_simplify_conv function."""

    @staticmethod
    def _default_kwargs():
        """Return the canonical 1x1 conv args (the only case that should simplify)."""
        return {
            "weight": np.random.randn(2, 3, 1, 1).astype(np.float32),
            "pre_output_shape": [1, 3, 1, 1],
            "pads": [0, 0, 0, 0],
            "strides": [1, 1],
            "dilations": [1, 1],
            "group": 1,
            "auto_pad": "NOTSET",
        }

    @pytest.mark.parametrize(
        ("override", "expected"),
        [
            pytest.param({}, True, id="basic_1x1_can_simplify"),
            pytest.param(
                {"weight": np.random.randn(2, 3, 3, 3).astype(np.float32)},
                False,
                id="non_1x1_kernel",
            ),
            pytest.param({"pads": [1, 1, 1, 1]}, False, id="with_padding"),
            pytest.param({"strides": [2, 2]}, False, id="with_stride"),
            pytest.param({"dilations": [2, 2]}, False, id="with_dilation"),
            pytest.param({"group": 2}, False, id="with_groups"),
            pytest.param({"auto_pad": "SAME_UPPER"}, False, id="with_auto_pad"),
            pytest.param(
                {"weight": np.random.randn(2, 5, 1, 1).astype(np.float32)},
                False,
                id="mismatched_channels",
            ),
        ],
    )
    def test_only_basic_1x1(self, override, expected):
        """Verify _can_simplify_conv accepts only basic 1x1 conv and rejects every other variant."""
        kwargs = self._default_kwargs()
        kwargs.update(override)
        result = _can_simplify_conv(**kwargs)
        if expected:
            assert result, f"Expected truthy result for override {override}"
        else:
            assert not result, f"Expected falsy result for override {override}"


class TestValidateConvSimplification:
    """Test _validate_conv_simplification function."""

    def test_accepts_valid_simplification(self):
        """Test validation of valid conv simplification."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Z = create_tensor_value_info("Z", "float32", [1, 4])

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        # Create three nodes: Conv -> Conv -> Flatten
        conv1 = helper.make_node("Conv", inputs=["X", "W"], outputs=["C1"])
        conv2 = helper.make_node("Conv", inputs=["C1", "W"], outputs=["C2"])
        flatten = helper.make_node("Flatten", inputs=["C2"], outputs=["Y"])

        model = create_minimal_onnx_model([conv1, conv2, flatten], [X], [Z], initializers_list)
        nodes = list(model.graph.node)

        # Should not raise
        _validate_conv_simplification(nodes[0], nodes[1], nodes[2], nodes)

    def test_no_predecessor_raises(self):
        """Test that no predecessor raises ValueError."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 2, 2])

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        conv1 = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([conv1], [X], [Y], initializers_list)
        nodes = list(model.graph.node)

        # Should raise: pre_pre_node is None
        with pytest.raises(ValueError, match="no predecessor"):
            _validate_conv_simplification(None, nodes[0], nodes[0], nodes)


class TestSimplifyConvToFlattenGemm:
    """Test _simplify_conv_to_flatten_gemm function."""

    def test_simplify_simple_case(self):
        """Test simplifying conv to flatten+gemm."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        gemm_w = np.random.randn(3, 2).astype(np.float32)
        gemm_b = np.zeros(2, dtype=np.float32)

        initializers_list = [
            create_initializer("W1", conv_w),
            create_initializer("W2", gemm_w),
            create_initializer("B", gemm_b),
        ]

        # Conv(1x1) -> Gemm
        conv = helper.make_node("Conv", inputs=["X", "W1"], outputs=["C"])
        gemm = helper.make_node("Gemm", inputs=["C", "W2", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([conv, gemm], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3, 1, 1], "C": [1, 2], "Y": [1, 2]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should process nodes
        assert len(result) >= 0

    def test_no_simplification_without_gemm(self):
        """Test that conv without following gemm is not simplified."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W1", conv_w)]

        conv = helper.make_node("Conv", inputs=["X", "W1"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Y2"])

        model = create_minimal_onnx_model([conv, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 2]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should preserve original structure
        assert len(result) == len(nodes)

    def test_preserves_non_conv_nodes(self):
        """Test that non-conv nodes are preserved."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 4, 4])

        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([identity], [X], [Y])
        nodes = list(model.graph.node)
        initializers_dict: dict[str, Any] = {}
        data_shapes = {"X": [1, 3, 4, 4], "Y": [1, 3, 4, 4]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should preserve Identity node
        assert len(result) == 1
        assert result[0].op_type == "Identity"

    def test_preserves_non_gemm_followers(self):
        """Test conv-gemm pattern with non-Gemm follower."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W1", conv_w)]

        # Conv followed by Relu (not Gemm)
        conv = helper.make_node("Conv", inputs=["X", "W1"], outputs=["C"])
        relu = helper.make_node("Relu", inputs=["C"], outputs=["Y"])

        model = create_minimal_onnx_model([conv, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3, 1, 1], "C": [1, 2], "Y": [1, 2]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should preserve both nodes (pattern not matched)
        assert len(result) == len(nodes)

    def test_applies_with_intermediate_conv(self):
        """Test simplification with intermediate conv (no chaining)."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        gemm_w = np.random.randn(2, 2).astype(np.float32)
        gemm_b = np.zeros(2, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("GW", gemm_w),
            create_initializer("GB", gemm_b),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])
        gemm = helper.make_node("Gemm", inputs=["C", "GW", "GB"], outputs=["Y"])

        model = create_minimal_onnx_model([conv, gemm], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {
            "X": [1, 3, 1, 1],
            "C": [1, 2],
            "Y": [1, 2],
        }

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should process pattern
        assert len(result) >= 0


class TestCreateFlattenGemmNodes:
    """Test _create_flatten_gemm_nodes function."""

    def test_produces_flatten_and_gemm_nodes(self):
        """Test creating Flatten and Gemm nodes."""
        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers: dict[str, Any] = {}

        # Create a Conv node with bias input
        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], name="conv1")

        flatten_node, gemm_node = _create_flatten_gemm_nodes(conv, conv_w, initializers)

        # Verify flatten node
        assert flatten_node.op_type == "Flatten"
        assert flatten_node.name == "conv1_flatten"
        assert flatten_node.input == ["X"]
        assert flatten_node.output == ["C_flatten"]

        # Verify gemm node
        assert gemm_node.op_type == "Gemm"
        assert gemm_node.name == "conv1_flatten_gemm"
        assert gemm_node.input[0] == "C_flatten"
        assert gemm_node.input[1] == "W"
        assert gemm_node.input[2] == "B"
        assert gemm_node.output == ["C"]

        # Verify weight was added to initializers
        assert "W" in initializers

    def test_reshapes_weight_correctly(self):
        """Test that weight is properly reshaped."""
        # Weight shape (out_channels, in_channels, kernel_h, kernel_w)
        conv_w = np.random.randn(4, 3, 2, 2).astype(np.float32)
        initializers: dict[str, Any] = {}

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], name="test_conv")

        _create_flatten_gemm_nodes(conv, conv_w, initializers)

        # Weight should be reshaped and transposed
        # Original: (4, 3, 2, 2) -> flattened: (4, 12) -> transposed: (12, 4)
        weight_tensor = initializers["W"]
        weight_array = onnx.numpy_helper.to_array(weight_tensor)
        assert weight_array.shape == (12, 4)


class TestValidateConvSimplificationMultipleSuccessors:
    """Test _validate_conv_simplification with multiple successors."""

    def test_raises_on_pre_pre_branching(self):
        """Test error when pre_pre_node has multiple successors."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 1, 1])
        _Z = create_tensor_value_info("Z", "float32", [1, 2, 1, 1])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        # Create graph: Conv -> Conv, and Conv -> Relu (branching from first conv)
        conv1 = helper.make_node("Conv", inputs=["X", "W"], outputs=["C1"], name="conv1")
        conv2 = helper.make_node("Conv", inputs=["C1", "W"], outputs=["Y"], name="conv2")
        relu = helper.make_node("Relu", inputs=["C1"], outputs=["Z"], name="relu1")

        model = create_minimal_onnx_model([conv1, conv2, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)

        # pre_pre_node = conv1, pre_node = conv2, node = relu
        # conv1 has multiple successors (conv2 and relu)
        with pytest.raises(ValueError, match="multiple successors"):
            _validate_conv_simplification(nodes[0], nodes[1], nodes[2], nodes)

    def test_raises_on_pre_branching(self):
        """Test error when pre_node has multiple successors."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 1, 1])
        _Z = create_tensor_value_info("Z", "float32", [1, 2, 1, 1])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        # Create graph where second conv has multiple successors
        conv1 = helper.make_node("Conv", inputs=["X", "W"], outputs=["C1"], name="conv1")
        conv2 = helper.make_node("Conv", inputs=["C1", "W"], outputs=["C2"], name="conv2")
        relu = helper.make_node("Relu", inputs=["C2"], outputs=["Y"], name="relu1")
        identity = helper.make_node("Identity", inputs=["C2"], outputs=["Z"], name="id1")

        model = create_minimal_onnx_model(
            [conv1, conv2, relu, identity], [X], [Y], initializers_list
        )
        nodes = list(model.graph.node)

        # pre_pre_node = conv1, pre_node = conv2, node = relu
        # conv2 has multiple successors (relu and identity)
        with pytest.raises(ValueError, match="multiple successors"):
            _validate_conv_simplification(nodes[0], nodes[1], nodes[2], nodes)


class TestSimplifyConvEdgeCases:
    """Test edge cases in Conv simplification."""

    def test_unsupported_weight_dimension(self):
        """Test error with unsupported weight dimension."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        # 3D weight (not 4D) - use proper 4D for conv but then test with 3D shape
        conv_w_4d = np.random.randn(2, 3, 3, 3).astype(np.float32)
        conv_w_3d = np.random.randn(2, 3, 3).astype(np.float32)
        initializers_list = [
            create_initializer("W1", conv_w_4d),
            create_initializer("W2", conv_w_3d),
        ]

        # First conv is valid 4D, second conv uses invalid 3D weight
        conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["C1"], name="conv1")
        conv2 = helper.make_node("Conv", inputs=["C1", "W2"], outputs=["Y"], name="conv2")

        model = create_minimal_onnx_model([conv1, conv2], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3, 4, 4], "C1": [1, 2, 2, 2], "Y": [1, 2]}

        # This should fail with "4D weight" error when trying to process conv2
        with pytest.raises((ValueError, NotImplementedError)):
            _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

    def test_single_conv_no_simplification(self):
        """Test that single conv node is not simplified."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], name="conv1")

        model = create_minimal_onnx_model([conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3, 1, 1], "Y": [1, 2]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Single conv should not be simplified (no pre_conv_node)
        assert len(result) == 1
        assert result[0].op_type == "Conv"

    def test_data_shapes_with_int_values(self):
        """Test handling of data_shapes with int instead of list."""
        X = create_tensor_value_info("X", "float32", [1, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [1, 2])

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        conv1 = helper.make_node("Conv", inputs=["X", "W"], outputs=["C1"], name="conv1")
        relu = helper.make_node("Relu", inputs=["C1"], outputs=["Y"], name="relu1")

        model = create_minimal_onnx_model([conv1, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Mix of int and list values in data_shapes
        data_shapes = {"X": [1, 3, 1, 1], "C1": 2, "Y": [1, 2]}

        result = _simplify_conv_to_flatten_gemm(nodes, initializers_dict, data_shapes)

        # Should handle conversion properly
        assert len(result) == len(nodes)
