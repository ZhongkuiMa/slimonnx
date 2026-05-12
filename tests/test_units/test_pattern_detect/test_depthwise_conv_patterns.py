"""Tests for depthwise convolution pattern detection."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
import pytest
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
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGetConvGroupAttr:
    """Test _get_conv_group_attr function."""

    @pytest.mark.parametrize(
        ("group_val", "weight_shape"),
        [
            (3, (3, 1, 3, 3)),
            (1, (3, 3, 3, 3)),
            (1, (1, 1, 3, 3)),
            (2, (2, 1, 3, 3)),
            (4, (4, 1, 3, 3)),
            (8, (8, 1, 3, 3)),
            (16, (16, 1, 3, 3)),
        ],
    )
    def test_extracts_group_attribute(self, group_val, weight_shape):
        """Test extracting group attribute from Conv nodes."""
        channels = weight_shape[0]
        X = create_tensor_value_info("X", "float32", [1, channels, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, channels, 32, 32])
        W = create_initializer("W", np.random.randn(*weight_shape).astype(np.float32))
        if group_val == 1:
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        else:
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group_val)
        model = create_minimal_onnx_model([conv], [X], [Y], [W])
        group = _get_conv_group_attr(next(iter(model.graph.node)))
        assert group == group_val


class TestIsDepthwiseConv:
    """Test _is_depthwise_conv function."""

    @pytest.mark.parametrize(
        ("op_type", "weight_shape", "group", "expected"),
        [
            pytest.param("Conv", (3, 1, 3, 3), 3, True, id="true_depthwise"),
            pytest.param("Conv", (3, 3, 3, 3), 1, False, id="false_group_one"),
            pytest.param("Conv", (4, 2, 3, 3), 2, False, id="false_wrong_shapes"),
            pytest.param("ConvTranspose", (3, 1, 3, 3), 3, True, id="true_convtranspose"),
        ],
    )
    def test_detects_depthwise_correctly(self, op_type, weight_shape, group, expected):
        """Test identification of depthwise vs regular convolution."""
        X = create_tensor_value_info("X", "float32", [1, weight_shape[0], 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, weight_shape[0], 32, 32])
        W = create_initializer("W", np.random.randn(*weight_shape).astype(np.float32))
        node = helper.make_node(op_type, inputs=["X", "W"], outputs=["Y"], group=group)
        model = create_minimal_onnx_model([node], [X], [Y], [W])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        assert _is_depthwise_conv(nodes[0], initializers) is expected

    @pytest.mark.parametrize(
        ("op_type", "has_weight"),
        [("Relu", True), ("Conv", False)],
    )
    def test_rejects_invalid_node_types(self, op_type, has_weight):
        """Test rejection of non-Conv nodes and missing weights."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
        if op_type == "Relu":
            node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
            model = create_minimal_onnx_model([node], [X], [Y])
        else:
            weight_name = "W" if has_weight else "missing_weight"
            conv = helper.make_node("Conv", inputs=["X", weight_name], outputs=["Y"], group=3)
            if has_weight:
                W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
                model = create_minimal_onnx_model([conv], [X], [Y], [W])
            else:
                model = create_minimal_onnx_model([conv], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        assert _is_depthwise_conv(nodes[0], initializers) is False


class TestIsConsecutiveNodes:
    """Test _is_consecutive_nodes function."""

    def test_true_when_output_matches_input(self):
        """Test identification of consecutive nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
        initializers_list = [
            W,
            create_initializer("bn_scale", np.ones(3, dtype=np.float32)),
            create_initializer("bn_bias", np.zeros(3, dtype=np.float32)),
            create_initializer("bn_mean", np.zeros(3, dtype=np.float32)),
            create_initializer("bn_var", np.ones(3, dtype=np.float32)),
        ]
        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["Y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Z"],
        )
        model = create_minimal_onnx_model([conv, bn], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        assert _is_consecutive_nodes(nodes[0], nodes[1], nodes) is True

    def test_false_when_outputs_differ(self):
        """Test non-consecutive when outputs don't match."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Z"], outputs=["Z"])
        model = create_minimal_onnx_model([conv, relu], [X], [Y], [W])
        nodes = list(model.graph.node)
        assert _is_consecutive_nodes(nodes[0], nodes[1], nodes) is False

    def test_false_when_multiple_consumers(self):
        """Test non-consecutive when multiple nodes consume output."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
        W_weight = create_initializer("W_weight", np.random.randn(3, 3, 3, 3).astype(np.float32))
        conv = helper.make_node("Conv", inputs=["X", "W_weight"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        identity = helper.make_node("Identity", inputs=["Y"], outputs=["W"])
        model = create_minimal_onnx_model([conv, relu, identity], [X], [Z], [W_weight])
        nodes = list(model.graph.node)
        assert _is_consecutive_nodes(nodes[0], nodes[1], nodes) is False


class TestDetectDepthwiseConv:
    """Test detect_depthwise_conv function."""

    @pytest.mark.parametrize(
        ("count", "mixed"),
        [(1, False), (2, False), (0, False), (1, True)],
    )
    def test_detects_depthwise_convs_in_graph(self, count, mixed):
        """Test detection of depthwise convolutions in various configurations."""
        if count == 0:
            # No depthwise conv
            X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
            Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
            W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
            model = create_minimal_onnx_model([conv], [X], [Y], [W])
        elif count == 1 and not mixed:
            # Single depthwise
            X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
            Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
            W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)
            model = create_minimal_onnx_model([conv], [X], [Y], [W])
        elif count == 2 and not mixed:
            # Multiple depthwise
            X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
            Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
            W1 = create_initializer("W1", np.random.randn(3, 1, 3, 3).astype(np.float32))
            W2 = create_initializer("W2", np.random.randn(3, 1, 3, 3).astype(np.float32))
            conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["Y"], group=3)
            conv2 = helper.make_node("Conv", inputs=["Y", "W2"], outputs=["Z"], group=3)
            model = create_minimal_onnx_model([conv1, conv2], [X], [Z], [W1, W2])
        else:
            # Mixed depthwise and regular
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
        assert len(result) == count


class TestDetectDepthwiseConvBn:
    """Test detect_depthwise_conv_bn function."""

    def _make_bn_initializers(self, channels: int = 3) -> list:
        """Build BN parameter initializers for the given channel count."""
        return [
            create_initializer("bn_scale", np.ones(channels, dtype=np.float32)),
            create_initializer("bn_bias", np.zeros(channels, dtype=np.float32)),
            create_initializer("bn_mean", np.zeros(channels, dtype=np.float32)),
            create_initializer("bn_var", np.ones(channels, dtype=np.float32)),
        ]

    def test_identifies_depthwise_conv_batch_norm_pattern(self):
        """Test detection of depthwise conv + BN pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
        initializers_list = [W, *self._make_bn_initializers()]
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

    def test_no_pattern_without_conv(self):
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

    def test_no_pattern_when_non_consecutive(self):
        """Test no detection when conv and BN not consecutive."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
        initializers_list = [W, *self._make_bn_initializers()]
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

    def test_no_pattern_when_missing_bn_params(self):
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

    def _make_bn_initializers(self, channels: int = 3) -> list:
        """Build BN parameter initializers for the given channel count."""
        return [
            create_initializer("bn_scale", np.ones(channels, dtype=np.float32)),
            create_initializer("bn_bias", np.zeros(channels, dtype=np.float32)),
            create_initializer("bn_mean", np.zeros(channels, dtype=np.float32)),
            create_initializer("bn_var", np.ones(channels, dtype=np.float32)),
        ]

    def test_identifies_batch_norm_depthwise_conv_pattern(self):
        """Test detection of BN + depthwise conv pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
        initializers_list = [W, *self._make_bn_initializers()]
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

    def test_no_pattern_without_conv(self):
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

    def test_no_pattern_when_non_consecutive(self):
        """Test no detection when BN and conv not consecutive."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])
        W = create_initializer("W", np.random.randn(3, 1, 3, 3).astype(np.float32))
        initializers_list = [W, *self._make_bn_initializers()]
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
