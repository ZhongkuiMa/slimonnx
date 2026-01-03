"""Extended tests for depthwise convolution fusion optimizations."""

from typing import Any

import numpy as np
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx._depthwise_conv import (
    _fuse_depthwise_conv_bn_or_bn_depthwise_conv,
    _get_conv_group_attr,
    _is_depthwise_conv,
)


def create_tensor_value_info(name, dtype, shape):
    """Create a TensorValueInfo."""
    return helper.make_tensor_value_info(name, dtype, shape)


def create_initializer(name, array):
    """Create an initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


class TestGetConvGroupAttr:
    """Test _get_conv_group_attr function."""

    def test_get_group_attr_default(self):
        """Test getting group attribute with default value."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"])

        group = _get_conv_group_attr(node)

        assert group == 1

    def test_get_group_attr_single(self):
        """Test getting group attribute with value 1."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=1)

        group = _get_conv_group_attr(node)

        assert group == 1

    def test_get_group_attr_depthwise(self):
        """Test getting group attribute for depthwise convolution."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        group = _get_conv_group_attr(node)

        assert group == 3

    def test_get_group_attr_large_group(self):
        """Test getting group attribute with large group value."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=64)

        group = _get_conv_group_attr(node)

        assert group == 64

    def test_get_group_attr_with_other_attrs(self):
        """Test getting group attribute with other attributes present."""
        node = helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["Y"], group=2, kernel_shape=[3, 3]
        )

        group = _get_conv_group_attr(node)

        assert group == 2


class TestIsDepthwiseConv:
    """Test _is_depthwise_conv function."""

    def test_is_depthwise_conv_true_basic(self):
        """Test depthwise conv detection - basic case."""
        # Create depthwise conv: group=3, out_channels=3, in_channels_per_group=1
        conv_weight = np.ones((3, 1, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3, name="conv_0")

        result = _is_depthwise_conv(node, initializers)

        assert result is True

    def test_is_depthwise_conv_true_large(self):
        """Test depthwise conv detection - large channels."""
        # Create depthwise conv: group=64, out_channels=64
        conv_weight = np.ones((64, 1, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=64)

        result = _is_depthwise_conv(node, initializers)

        assert result is True

    def test_is_depthwise_conv_false_group_one(self):
        """Test depthwise conv detection - group=1 is not depthwise."""
        conv_weight = np.ones((3, 3, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=1)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_is_depthwise_conv_false_regular_grouped(self):
        """Test depthwise conv detection - regular grouped conv (not depthwise)."""
        # Regular grouped conv: group=2, out_channels=4, in_channels_per_group=2
        conv_weight = np.ones((4, 2, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=2)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_is_depthwise_conv_false_non_conv(self):
        """Test depthwise conv detection - non-Conv node."""
        initializers: dict[str, Any] = {}

        node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_is_depthwise_conv_false_no_weight(self):
        """Test depthwise conv detection - Conv without weight."""
        initializers: dict[str, Any] = {}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_is_depthwise_conv_false_missing_input(self):
        """Test depthwise conv detection - Conv with insufficient inputs."""
        conv_weight = np.ones((3, 1, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X"], outputs=["Y"], group=3)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_is_depthwise_conv_false_weight_shape(self):
        """Test depthwise conv detection - weight shape mismatch."""
        # Weight shape doesn't match: in_channels_per_group != 1
        conv_weight = np.ones((3, 2, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=3)

        result = _is_depthwise_conv(node, initializers)

        assert result is False


class TestFuseDepthwiseConvBn:
    """Test _fuse_depthwise_conv_bn_or_bn_depthwise_conv function."""

    def test_fuse_depthwise_conv_bn_basic(self):
        """Test depthwise Conv+BN fusion - basic case."""
        # Setup depthwise conv
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.zeros(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should have fused into single node
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"
        assert result_nodes[0].output[0] == "Y"

    def test_fuse_depthwise_conv_bn_multiple_nodes(self):
        """Test depthwise Conv+BN fusion with non-matching nodes."""
        # Setup depthwise conv
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.zeros(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes with intermediate non-matching node
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"])
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["relu_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, relu_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should not fuse - BN not immediately after Conv
        assert len(result_nodes) == 3

    def test_fuse_bn_depthwise_conv_basic(self):
        """Test BN+depthwise Conv fusion - basic case."""
        # Setup BN
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        # Setup depthwise conv
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.zeros(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        initializers = {
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
            "W": weight_init,
            "B": bias_init,
        }

        # Create nodes
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            name="bn_0",
        )
        conv_node = helper.make_node(
            "Conv", inputs=["bn_out", "W", "B"], outputs=["Y"], group=3, name="conv_0"
        )

        nodes = [bn_node, conv_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=False
        )

        # Should have fused into single node
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"
        assert result_nodes[0].output[0] == "Y"

    def test_fuse_depthwise_conv_bn_without_bias(self):
        """Test depthwise Conv+BN fusion without Conv bias."""
        # Setup depthwise conv without bias
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        weight_init = create_initializer("W", conv_weight)

        # Setup BN
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes - Conv without bias
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should have fused
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_fuse_non_depthwise_conv(self):
        """Test fusion skips non-depthwise Conv nodes."""
        # Setup regular (non-depthwise) conv
        conv_weight = np.random.randn(3, 3, 3, 3).astype(np.float32)  # Not depthwise
        conv_bias = np.zeros(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes with non-depthwise Conv
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=1, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should not fuse - not depthwise
        assert len(result_nodes) == 2

    def test_fuse_depthwise_conv_bn_different_scales(self):
        """Test depthwise Conv+BN fusion with varied BN scales."""
        # Setup depthwise conv
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.ones(3, dtype=np.float32) * 0.5
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN with different scales
        bn_scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        bn_bias = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        bn_mean = np.array([0.0, 0.1, -0.1], dtype=np.float32)
        bn_var = np.array([1.0, 2.0, 0.5], dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should have fused
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_fuse_depthwise_conv_bn_large_kernels(self):
        """Test depthwise Conv+BN fusion with large kernels."""
        # Setup depthwise conv with large kernel
        conv_weight = np.random.randn(16, 1, 5, 5).astype(np.float32)
        conv_bias = np.zeros(16, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN
        bn_scale = np.ones(16, dtype=np.float32)
        bn_bias = np.zeros(16, dtype=np.float32)
        bn_mean = np.zeros(16, dtype=np.float32)
        bn_var = np.ones(16, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Create nodes
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=16, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Should have fused
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_fuse_empty_node_list(self):
        """Test fusion with empty node list."""
        initializers: dict[str, Any] = {}
        nodes: list[Any] = []

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 0

    def test_fuse_single_node(self):
        """Test fusion with single node."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        conv_node = helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["Y"], group=3, name="conv_0"
        )

        nodes = [conv_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 1

    def test_fuse_depthwise_conv_bn_dtype_preservation_float32(self):
        """Test that float32 dtype is preserved in fusion (lines 130-133)."""
        # Create depthwise conv with float32
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.ones(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # Setup BN with float32
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers, is_conv_bn=True)

        # Verify dtype preservation
        assert "W" in initializers
        fused_weight = numpy_helper.to_array(initializers["W"])
        assert fused_weight.dtype == np.float32

    def test_fuse_depthwise_conv_bn_fusion_formula_conv_bn(self):
        """Test Conv+BN fusion formula (lines 138-141)."""
        # Create simple depthwise conv: group=3, out_channels=3, in_channels_per_group=1
        conv_weight = np.ones((3, 1, 1, 1), dtype=np.float32)  # [3, 1, 1, 1]
        conv_bias = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # BN parameters
        bn_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        bn_bias = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        bn_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bn_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Depthwise conv with group=3
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Verify fusion result
        assert len(result_nodes) == 1
        fused_weight = numpy_helper.to_array(initializers["W"])
        fused_bias = numpy_helper.to_array(initializers["B"])

        # Formula: new_weight = weight * bn_weight, new_bias = bias * bn_weight + bn_bias
        # new_weight = 1.0 * 2.0 = 2.0
        # new_bias = 2.0 * 2.0 + 1.0 = 5.0
        assert np.allclose(fused_weight, 2.0)
        assert np.allclose(fused_bias, 5.0)

    def test_fuse_depthwise_conv_bn_fusion_formula_bn_conv(self):
        """Test BN+Conv fusion formula (lines 143-146)."""
        # Create simple depthwise conv: group=3, out_channels=3, in_channels_per_group=1
        conv_weight = np.ones((3, 1, 1, 1), dtype=np.float32)  # [3, 1, 1, 1]
        conv_bias = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        # BN parameters
        bn_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        bn_bias = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        bn_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bn_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            name="bn_0",
        )
        # Depthwise conv with group=3
        conv_node = helper.make_node(
            "Conv", inputs=["bn_out", "W", "B"], outputs=["Y"], group=3, name="conv_0"
        )

        nodes = [bn_node, conv_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=False
        )

        # Verify fusion result
        assert len(result_nodes) == 1
        fused_weight = numpy_helper.to_array(initializers["W"])
        fused_bias = numpy_helper.to_array(initializers["B"])

        # Formula: new_weight = weight * bn_weight, new_bias = bias + bn_bias
        # new_weight = 1.0 * 2.0 = 2.0
        # new_bias = 2.0 + 1.0 = 3.0
        assert np.allclose(fused_weight, 2.0)
        assert np.allclose(fused_bias, 3.0)

    def test_fuse_depthwise_conv_bn_bias_name_generation(self):
        """Test bias name generation (lines 150-153)."""
        # Test case: Conv without bias - should generate name
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        weight_init = create_initializer("W", conv_weight)

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Conv without bias (only 2 inputs)
        conv_node = helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Verify generated bias name exists
        assert len(result_nodes) == 1
        fused_node = result_nodes[0]
        # New bias name should be created
        assert len(fused_node.input) == 3
        bias_name = fused_node.input[2]
        assert bias_name in initializers
        assert "_bias" in bias_name

    def test_fuse_depthwise_conv_bn_input_output_wiring_conv_bn(self):
        """Test input/output wiring for Conv+BN (lines 166-168)."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.ones(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # Conv+BN with specific input/output names
        conv_node = helper.make_node(
            "Conv", inputs=["input_X", "W", "B"], outputs=["conv_out"], group=3, name="conv_0"
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["final_output"],
            name="bn_0",
        )

        nodes = [conv_node, bn_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        # Verify input/output wiring
        assert len(result_nodes) == 1
        fused_node = result_nodes[0]
        # Input should come from Conv input[0]
        assert fused_node.input[0] == "input_X"
        # Output should be BN output
        assert fused_node.output[0] == "final_output"

    def test_fuse_depthwise_conv_bn_input_output_wiring_bn_conv(self):
        """Test input/output wiring for BN+Conv (lines 170-171)."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.ones(3, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        bn_scale_init = create_initializer("bn_scale", bn_scale)
        bn_bias_init = create_initializer("bn_bias", bn_bias)
        bn_mean_init = create_initializer("bn_mean", bn_mean)
        bn_var_init = create_initializer("bn_var", bn_var)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": bn_scale_init,
            "bn_bias": bn_bias_init,
            "bn_mean": bn_mean_init,
            "bn_var": bn_var_init,
        }

        # BN+Conv with specific input/output names
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["input_X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            name="bn_0",
        )
        conv_node = helper.make_node(
            "Conv", inputs=["bn_out", "W", "B"], outputs=["final_output"], group=3, name="conv_0"
        )

        nodes = [bn_node, conv_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=False
        )

        # Verify input/output wiring
        assert len(result_nodes) == 1
        fused_node = result_nodes[0]
        # Input should come from BN input[0]
        assert fused_node.input[0] == "input_X"
        # Output should be Conv output
        assert fused_node.output[0] == "final_output"
