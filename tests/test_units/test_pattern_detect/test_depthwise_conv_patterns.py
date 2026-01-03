"""Tests for depthwise convolution pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.depthwise_conv import (
    _get_conv_group_attr,
    _is_depthwise_conv,
    detect_bn_depthwise_conv,
    detect_depthwise_conv,
    detect_depthwise_conv_bn,
)
from slimonnx.pattern_detect.utils import is_consecutive_nodes as _is_consecutive_nodes

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGetConvGroupAttr:
    """Test _get_conv_group_attr function."""

    def test_get_group_attribute_present(self):
        """Test extracting group attribute when present."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)

        group = _get_conv_group_attr(nodes[0])
        assert group == 3

    def test_get_group_attribute_default(self):
        """Test group attribute defaults to 1 when not specified."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)

        group = _get_conv_group_attr(nodes[0])
        assert group == 1

    def test_get_group_attribute_various_values(self):
        """Test group attribute with various values."""
        for group_val in [1, 2, 4, 8, 16]:
            X = create_tensor_value_info("X", "float32", [1, 16, 32, 32])
            Y = create_tensor_value_info("Y", "float32", [1, 16, 32, 32])

            W = create_initializer("W", np.random.randn(16, 1, 3, 3).astype(np.float32))

            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group_val)

            model = create_minimal_onnx_model([conv], [X], [Y], [W])
            nodes = list(model.graph.node)

            group = _get_conv_group_attr(nodes[0])
            assert group == group_val


class TestIsDepthwiseConv:
    """Test _is_depthwise_conv function."""

    def test_is_depthwise_conv_true(self):
        """Test identification of depthwise convolution."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is True

    def test_is_depthwise_conv_false_group_one(self):
        """Test non-depthwise with group=1."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is False

    def test_is_depthwise_conv_false_wrong_shapes(self):
        """Test non-depthwise when channel counts don't match."""
        X = create_tensor_value_info("X", "float32", [1, 4, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 4, 32, 32])

        W = create_initializer("W", np.random.randn(4, 2, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=2)

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is False

    def test_is_depthwise_conv_non_conv_node(self):
        """Test with non-Conv node."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is False

    def test_is_depthwise_conv_missing_weight(self):
        """Test with missing weight tensor."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        conv = helper.make_node("Conv", inputs=["X", "missing_weight"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is False

    def test_is_depthwise_convtranspose(self):
        """Test depthwise ConvTranspose detection."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 64, 64])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))

        convt = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([convt], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        is_depthwise = _is_depthwise_conv(nodes[0], initializers)
        assert is_depthwise is True


class TestIsConsecutiveNodes:
    """Test _is_consecutive_nodes function."""

    def test_is_consecutive_nodes_true(self):
        """Test identification of consecutive nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["Y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Z], initializers_list)
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is True

    def test_is_consecutive_nodes_false_different_outputs(self):
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

    def test_is_consecutive_nodes_false_multiple_consumers(self):
        """Test non-consecutive when multiple nodes consume output."""
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


class TestDetectDepthwiseConv:
    """Test detect_depthwise_conv function."""

    def test_detect_depthwise_conv_single(self):
        """Test detection of single depthwise convolution."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["op_type"] == "Conv"
        assert result[0]["group"] == 3

    def test_detect_depthwise_conv_multiple(self):
        """Test detection of multiple depthwise convolutions."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W1 = create_initializer("W1", np.random.randn(3, 1, 3, 3).astype(np.float32))
        W2 = create_initializer("W2", np.random.randn(3, 1, 3, 3).astype(np.float32))

        conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["Y"], group=3)
        conv2 = helper.make_node("Conv", inputs=["Y", "W2"], outputs=["Z"], group=3)

        model = create_minimal_onnx_model([conv1, conv2], [X], [Z], [W1, W2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_detect_depthwise_conv_none(self):
        """Test no detection when no depthwise conv present."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_depthwise_conv_with_non_depthwise(self):
        """Test detection with mixed depthwise and regular convolutions."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W1 = create_initializer("W1", np.random.randn(3, 1, 3, 3).astype(np.float32))
        W2 = create_initializer("W2", np.random.randn(3, 3, 3, 3).astype(np.float32))

        conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["Y"], group=3)
        conv2 = helper.make_node("Conv", inputs=["Y", "W2"], outputs=["Z"])

        model = create_minimal_onnx_model([conv1, conv2], [X], [Z], [W1, W2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1


class TestDetectDepthwiseConvBn:
    """Test detect_depthwise_conv_bn function."""

    def test_detect_depthwise_conv_bn_pattern(self):
        """Test detection of depthwise conv + BN pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["conv_node"] == conv.name
        assert result[0]["bn_node"] == bn.name

    def test_detect_depthwise_conv_bn_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_depthwise_conv_bn_non_consecutive(self):
        """Test no detection when conv and BN not consecutive."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"], group=3)
        relu = helper.make_node("Relu", inputs=["C"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv, relu, bn], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_depthwise_conv_bn_missing_bn_params(self):
        """Test no detection when BN missing parameters."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_depthwise_conv_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0


class TestDetectBnDepthwiseConv:
    """Test detect_bn_depthwise_conv function."""

    def test_detect_bn_depthwise_conv_pattern(self):
        """Test detection of BN + depthwise conv pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
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
        conv = helper.make_node("Conv", inputs=["B", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["bn_node"] == bn.name
        assert result[0]["conv_node"] == conv.name

    def test_detect_bn_depthwise_conv_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_bn_depthwise_conv_non_consecutive(self):
        """Test no detection when BN and conv not consecutive."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
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
        relu = helper.make_node("Relu", inputs=["B"], outputs=["R"])
        conv = helper.make_node("Conv", inputs=["R", "W"], outputs=["Z"], group=3)

        model = create_minimal_onnx_model([bn, relu, conv], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_depthwise_conv(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0
