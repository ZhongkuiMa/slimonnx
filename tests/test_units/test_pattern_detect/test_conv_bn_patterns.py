"""Tests for Conv + BatchNorm pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.conv_bn import (
    detect_bn_conv,
    detect_bn_conv_transpose,
    detect_conv_bn,
    detect_conv_transpose_bn,
)
from slimonnx.pattern_detect.utils import is_consecutive_nodes as _is_consecutive_nodes

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestIsConsecutiveNodes:
    """Test _is_consecutive_nodes function for Conv+BN patterns."""

    def test_is_consecutive_conv_bn(self):
        """Test consecutive Conv->BN detection."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["Y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is True

    def test_is_consecutive_false_different_outputs(self):
        """Test non-consecutive when outputs don't match."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["Z"])

        model = create_minimal_onnx_model([conv, relu], [X], [Y], [W])
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is False

    def test_is_consecutive_false_multiple_consumers(self):
        """Test non-consecutive when multiple consumers."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W_weight = create_initializer("W_weight", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W_weight"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        identity = helper.make_node("Identity", inputs=["Y"], outputs=["W"])

        model = create_minimal_onnx_model([conv, relu, identity], [X], [Z], [W_weight])
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is False


class TestDetectConvBn:
    """Test detect_conv_bn function."""

    def test_detect_conv_bn_pattern(self):
        """Test detection of Conv+BN pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["conv_node"] == conv.name
        assert result[0]["bn_node"] == bn.name
        assert result[0]["can_fuse"] is True

    def test_detect_conv_bn_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_conv_bn_non_consecutive(self):
        """Test no detection when Conv and BN not consecutive."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])
        relu = helper.make_node("Relu", inputs=["C"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv, relu, bn], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_conv_bn_missing_conv_weight(self):
        """Test no detection when Conv weight missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "missing_weight"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_conv_bn_missing_bn_params(self):
        """Test no detection when BN params missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_conv_bn_multiple_patterns(self):
        """Test detection of multiple Conv+BN patterns."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W1 = create_initializer("W1", np.random.randn(3, 3, 3, 3).astype(np.float32))
        W2 = create_initializer("W2", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W1,
            W2,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["C1"])
        bn1 = helper.make_node(
            "BatchNormalization",
            inputs=["C1", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B1"],
        )
        conv2 = helper.make_node("Conv", inputs=["B1", "W2"], outputs=["C2"])
        bn2 = helper.make_node(
            "BatchNormalization",
            inputs=["C2", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv1, bn1, conv2, bn2], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 2


class TestDetectBnConv:
    """Test detect_bn_conv function."""

    def test_detect_bn_conv_pattern(self):
        """Test detection of BN+Conv pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B"],
        )
        conv = helper.make_node("Conv", inputs=["B", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["bn_node"] == bn.name
        assert result[0]["conv_node"] == conv.name

    def test_detect_bn_conv_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_bn_conv_missing_bn_params(self):
        """Test no detection when BN params missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B"],
        )
        conv = helper.make_node("Conv", inputs=["B", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_bn_conv_missing_conv_weight(self):
        """Test no detection when Conv weight missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B"],
        )
        conv = helper.make_node("Conv", inputs=["B", "missing_weight"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0


class TestDetectConvTransposeBn:
    """Test detect_conv_transpose_bn function."""

    def test_detect_conv_transpose_bn_pattern(self):
        """Test detection of ConvTranspose+BN pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 64, 64])

        W = create_initializer("W", np.random.randn(3, 2, 3, 3).astype(np.float32))
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        convt = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([convt, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_transpose_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["conv_transpose_node"] == convt.name

    def test_detect_conv_transpose_bn_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_conv_transpose_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0


class TestDetectBnConvTranspose:
    """Test detect_bn_conv_transpose function."""

    def test_detect_bn_conv_transpose_pattern(self):
        """Test detection of BN+ConvTranspose pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 64, 64])

        W = create_initializer("W", np.random.randn(3, 2, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B"],
        )
        convt = helper.make_node("ConvTranspose", inputs=["B", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, convt], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv_transpose(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["bn_node"] == bn.name

    def test_detect_bn_conv_transpose_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_conv_transpose(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0
