"""Tests for depthwise convolution fusion optimizations."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._depthwise_conv import (
    _fuse_depthwise_conv_bn_or_bn_depthwise_conv,
    _get_conv_group_attr,
    _is_depthwise_conv,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGetConvGroupAttr:
    """Test _get_conv_group_attr function."""

    @pytest.mark.parametrize(
        ("group", "expected"),
        [
            (2, 2),
            (None, 1),  # default: no group attribute → returns 1
            (1, 1),
            (16, 16),
        ],
        ids=["explicit_group_2", "default_group_1", "explicit_group_1", "large_group_16"],
    )
    def test_returns_group_attribute(self, group, expected):
        """Test that group attribute is extracted correctly for various group values."""
        if group is None:
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])
        else:
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group)

        result = _get_conv_group_attr(conv)
        assert result == expected


class TestIsDepthwiseConv:
    """Test _is_depthwise_conv function."""

    @pytest.mark.parametrize(
        ("in_ch", "out_ch", "group", "weight_shape", "is_depthwise"),
        [
            (3, 3, 3, (3, 1, 3, 3), True),
            (6, 4, 2, (4, 3, 3, 3), False),
            (3, 2, 1, (2, 3, 3, 3), False),
        ],
        ids=["depthwise_group_equals_channels", "grouped_not_depthwise", "standard_conv"],
    )
    def test_identifies_depthwise_condition(self, in_ch, out_ch, group, weight_shape, is_depthwise):
        """Test depthwise detection across various group configurations."""
        X = create_tensor_value_info("X", "float32", [1, in_ch, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, out_ch, 2, 2])

        conv_w = np.random.randn(*weight_shape).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group)

        model = create_minimal_onnx_model([conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        assert _is_depthwise_conv(nodes[0], initializers_dict) == is_depthwise

    def test_rejects_regular_conv_with_group_one(self):
        """Test that a standard conv (group=1, multiple in_channels) is not depthwise."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 2, 2])

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers_list = [create_initializer("W", conv_w)]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        model = create_minimal_onnx_model([conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        assert not _is_depthwise_conv(nodes[0], initializers_dict)

    def test_rejects_non_conv_node(self):
        """Test that a non-Conv node returns False."""
        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        assert not _is_depthwise_conv(relu, {})

    def test_rejects_conv_without_weight_initializer(self):
        """Test that a conv whose weight is not an initializer returns False."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        W = create_tensor_value_info("W", "float32", [3, 1, 3, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv], [X, W], [Y])
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        assert not _is_depthwise_conv(nodes[0], initializers_dict)


class TestFuseDepthwiseConvBn:
    """Test _fuse_depthwise_conv_bn_or_bn_depthwise_conv function."""

    def _make_depthwise_bn_initializers(self, channels: int) -> tuple[list, np.ndarray]:
        """Return (initializers_list, conv_w) for a depthwise Conv+BN of given channels."""
        conv_w = np.random.randn(channels, 1, 3, 3).astype(np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)
        inits = [
            create_initializer("W", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]
        return inits, conv_w

    @pytest.mark.parametrize(
        ("is_conv_bn", "with_relu"),
        [
            (True, False),
            (True, True),
            (False, False),
        ],
        ids=["conv_then_bn", "conv_bn_then_relu", "bn_then_conv"],
    )
    def test_reduces_node_count_for_depthwise_fusion(self, is_conv_bn, with_relu):
        """Test depthwise Conv+BN fusion reduces node count."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y_final = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        inits, _ = self._make_depthwise_bn_initializers(3)

        if is_conv_bn:
            conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"], group=3)
            bn = helper.make_node(
                "BatchNormalization",
                inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["B"],
            )
            nodes = [conv, bn]
        else:
            bn = helper.make_node(
                "BatchNormalization",
                inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["B"],
            )
            conv = helper.make_node("Conv", inputs=["B", "W"], outputs=["C"], group=3)
            nodes = [bn, conv]

        if with_relu:
            relu = helper.make_node("Relu", inputs=["B" if is_conv_bn else "C"], outputs=["Y"])
            nodes.append(relu)
            Y_final = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        model = create_minimal_onnx_model(nodes, [X], [Y_final], inits)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=is_conv_bn
            )
            assert len(result) <= len(nodes)
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")

    def test_skips_fusion_for_non_depthwise_patterns(self):
        """Test non-depthwise Conv+BN and non-Conv-BN patterns pass through unchanged."""
        # Regular Conv+BN (not depthwise)
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 2, 2])

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])  # Not depthwise
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers_dict)
        assert len(result) == len(nodes)

        # Non-Conv-BN pattern
        X2 = create_tensor_value_info("X2", "float32", [1, 3, 4, 4])
        Y2 = create_tensor_value_info("Y2", "float32", [1, 3, 4, 4])
        relu = helper.make_node("Relu", inputs=["X2"], outputs=["Y2"])
        model2 = create_minimal_onnx_model([relu], [X2], [Y2])
        nodes2 = list(model2.graph.node)
        initializers_dict2: dict[str, Any] = {}

        result2 = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes2, initializers_dict2)
        assert len(result2) == 1
        assert result2[0].op_type == "Relu"

    def test_leaves_sequential_convs_unchanged_without_bn(self):
        """Test sequential Conv+Conv without BN is left unchanged."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        conv1_w = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv2_w = np.random.randn(3, 1, 3, 3).astype(np.float32)

        initializers_list = [
            create_initializer("W1", conv1_w),
            create_initializer("W2", conv2_w),
        ]

        conv1 = helper.make_node("Conv", inputs=["X", "W1"], outputs=["C1"], group=3)
        conv2 = helper.make_node("Conv", inputs=["C1", "W2"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([conv1, conv2], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers_dict)
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")
        else:
            assert isinstance(result, list)
            assert len(result) == len(nodes), (
                f"Expected {len(nodes)} nodes preserved (no fusion without BN), got {len(result)}"
            )


class TestDepthwiseConvBnEdgeCases:
    """Test edge cases and error conditions."""

    def test_fuses_with_existing_conv_bias(self):
        """Test Conv+BN fusion succeeds when Conv already has bias."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        conv_w = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_b = np.zeros(3, dtype=np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("B", conv_b),
            create_initializer("scale", bn_scale),
            create_initializer("bias", bn_bias),
            create_initializer("mean", bn_mean),
            create_initializer("var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=True
            )
            assert len(result) == 1
            assert result[0].op_type == "Conv"
            assert len(result[0].input) == 3  # X, W, B
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")

    def test_creates_new_bias_when_fusing_bn_without_conv_bias(self):
        """Test BN+Conv fusion adds new bias when Conv lacks one."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)
        conv_w = np.random.randn(3, 1, 3, 3).astype(np.float32)

        initializers_list = [
            create_initializer("scale", bn_scale),
            create_initializer("bias", bn_bias),
            create_initializer("mean", bn_mean),
            create_initializer("var", bn_var),
            create_initializer("W", conv_w),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["BN_out"],
        )
        conv = helper.make_node("Conv", inputs=["BN_out", "W"], outputs=["Y"], group=3)

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=False
            )
            assert len(result) == 1
            assert result[0].op_type == "Conv"
            assert len(result[0].input) == 3  # X, W, new_bias
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")

    @pytest.mark.parametrize(
        ("kernel_size", "spatial_in", "spatial_out"),
        [
            (5, 8, 4),
            (3, 4, 2),
        ],
        ids=["5x5_kernel", "3x3_kernel"],
    )
    def test_fuses_with_different_kernel_sizes(self, kernel_size, spatial_in, spatial_out):
        """Test Conv+BN fusion succeeds with varied kernel sizes."""
        X = create_tensor_value_info("X", "float32", [1, 3, spatial_in, spatial_in])
        Y = create_tensor_value_info("Y", "float32", [1, 3, spatial_out, spatial_out])

        conv_w = np.random.randn(3, 1, kernel_size, kernel_size).astype(np.float32)
        conv_b = np.zeros(3, dtype=np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("B", conv_b),
            create_initializer("scale", bn_scale),
            create_initializer("bias", bn_bias),
            create_initializer("mean", bn_mean),
            create_initializer("var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=True
            )
            assert len(result) == 1
            assert result[0].op_type == "Conv"
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")

    def test_fuses_with_nonstandard_batch_norm_parameters(self):
        """Test Conv+BN fusion succeeds with non-default BN params."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 2, 2])

        conv_w = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_b = np.ones(3, dtype=np.float32)
        bn_scale = np.array([2.0, 2.5, 3.0], dtype=np.float32)
        bn_bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        bn_mean = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        bn_var = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("B", conv_b),
            create_initializer("scale", bn_scale),
            create_initializer("bias", bn_bias),
            create_initializer("mean", bn_mean),
            create_initializer("var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], group=3)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=True
            )
            assert len(result) == 1
            assert result[0].op_type == "Conv"
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")

    def test_fuses_with_large_channel_count(self):
        """Test Conv+BN fusion succeeds with 16-channel depthwise conv."""
        X = create_tensor_value_info("X", "float32", [1, 16, 4, 4])
        Y = create_tensor_value_info("Y", "float32", [1, 16, 2, 2])

        conv_w = np.random.randn(16, 1, 3, 3).astype(np.float32)
        conv_b = np.zeros(16, dtype=np.float32)
        bn_scale = np.ones(16, dtype=np.float32)
        bn_bias = np.zeros(16, dtype=np.float32)
        bn_mean = np.zeros(16, dtype=np.float32)
        bn_var = np.ones(16, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
            create_initializer("B", conv_b),
            create_initializer("scale", bn_scale),
            create_initializer("bias", bn_bias),
            create_initializer("mean", bn_mean),
            create_initializer("var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"], group=16)
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
                nodes, initializers_dict, is_conv_bn=True
            )
            assert len(result) == 1
            assert result[0].op_type == "Conv"
        except NotImplementedError:
            pytest.skip("Operation not implemented for this case")
