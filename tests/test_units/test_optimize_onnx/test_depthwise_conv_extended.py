"""Extended tests for depthwise convolution fusion optimizations."""

__docformat__ = "restructuredtext"

from typing import Any

import numpy as np
import pytest
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


def _build_conv_bn_initializers(channels, conv_weight, has_bias=True):
    """Build standard depthwise Conv+BN initializer dict."""
    inits = {"W": create_initializer("W", conv_weight)}
    if has_bias:
        inits["B"] = create_initializer("B", np.zeros(channels, dtype=np.float32))
    inits["bn_scale"] = create_initializer("bn_scale", np.ones(channels, dtype=np.float32))
    inits["bn_bias"] = create_initializer("bn_bias", np.zeros(channels, dtype=np.float32))
    inits["bn_mean"] = create_initializer("bn_mean", np.zeros(channels, dtype=np.float32))
    inits["bn_var"] = create_initializer("bn_var", np.ones(channels, dtype=np.float32))
    return inits


def _build_conv_bn_nodes(channels, has_bias=True, is_conv_bn=True):
    """Build standard depthwise Conv+BN node list."""
    if has_bias:
        conv_inputs = ["X", "W", "B"]
    else:
        conv_inputs = ["X", "W"]
    conv_node = helper.make_node(
        "Conv", inputs=conv_inputs, outputs=["conv_out"], group=channels, name="conv_0"
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["Y"],
        name="bn_0",
    )
    if is_conv_bn:
        return [conv_node, bn_node]
    # BN+Conv order
    bn_node_first = helper.make_node(
        "BatchNormalization",
        inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["bn_out"],
        name="bn_0",
    )
    conv_node_second = helper.make_node(
        "Conv", inputs=["bn_out", "W", "B"], outputs=["Y"], group=channels, name="conv_0"
    )
    return [bn_node_first, conv_node_second]


class TestGetConvGroupAttr:
    """Test _get_conv_group_attr function."""

    @pytest.mark.parametrize(
        ("group_kwargs", "expected_group"),
        [
            ({}, 1),
            ({"group": 1}, 1),
            ({"group": 3}, 3),
            ({"group": 64}, 64),
        ],
    )
    def test_returns_correct_group_value(self, group_kwargs, expected_group):
        """Test getting group attribute returns correct value for various inputs."""
        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], **group_kwargs)

        group = _get_conv_group_attr(node)

        assert group == expected_group

    def test_extracts_with_multiple_attributes_present(self):
        """Test getting group attribute when other Conv attributes present."""
        node = helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["Y"], group=2, kernel_shape=[3, 3]
        )

        group = _get_conv_group_attr(node)

        assert group == 2


class TestIsDepthwiseConv:
    """Test _is_depthwise_conv function."""

    @pytest.mark.parametrize(
        ("weight_shape", "group"),
        [
            ((3, 1, 3, 3), 3),
            ((64, 1, 3, 3), 64),
        ],
    )
    def test_returns_true_for_valid_configurations(self, weight_shape, group):
        """Test depthwise conv detection returns True for valid configurations."""
        conv_weight = np.ones(weight_shape, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group)

        result = _is_depthwise_conv(node, initializers)

        assert result is True

    def test_returns_false_for_group_one(self):
        """Test depthwise conv detection - group=1 is not depthwise."""
        conv_weight = np.ones((3, 3, 3, 3), dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=1)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    @pytest.mark.parametrize(
        ("weight_shape", "group"),
        [
            pytest.param((4, 2, 3, 3), 2, id="regular_grouped_conv"),
            pytest.param((3, 2, 3, 3), 3, id="weight_shape_mismatch"),
        ],
    )
    def test_returns_false_for_non_depthwise_conv(self, weight_shape, group):
        """Test depthwise conv detection returns False for non-depthwise configurations."""
        conv_weight = np.ones(weight_shape, dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        initializers = {"W": weight_init}

        node = helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], group=group)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    @pytest.mark.parametrize(
        ("op_type", "inputs", "group_kwargs", "init_dict"),
        [
            ("Add", ["X", "Y"], {}, {}),
            ("Conv", ["X", "W"], {"group": 3}, {}),
            ("Conv", ["X"], {"group": 3}, {"W": None}),
        ],
    )
    def test_returns_false_for_non_depthwise_configurations(
        self, op_type, inputs, group_kwargs, init_dict
    ):
        """Test depthwise conv detection returns False for non-Conv, missing weight, or bad input count."""
        if op_type == "Conv" and "W" in init_dict and init_dict["W"] is None:
            conv_weight = np.ones((3, 1, 3, 3), dtype=np.float32)
            weight_init = create_initializer("W", conv_weight)
            initializers = {"W": weight_init}
        else:
            initializers: dict[str, Any] = dict(init_dict)

        node = helper.make_node(op_type, inputs=inputs, outputs=["Z"], **group_kwargs)

        result = _is_depthwise_conv(node, initializers)

        assert result is False

    def test_returns_false_for_weight_shape_mismatch(self):
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

    @pytest.mark.parametrize(
        "is_conv_bn",
        [
            pytest.param(True, id="conv_then_bn"),
            pytest.param(False, id="bn_then_conv"),
        ],
    )
    def test_fuses_into_single_conv_node(self, is_conv_bn):
        """Test depthwise Conv+BN or BN+Conv fusion results in single Conv node."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=True)
        nodes = _build_conv_bn_nodes(3, has_bias=True, is_conv_bn=is_conv_bn)

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=is_conv_bn
        )

        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"
        assert result_nodes[0].output[0] == "Y"

    def test_skips_fusion_when_relu_between_operations(self):
        """Test fusion is skipped when ReLU separates Conv and BN."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=True)

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

    def test_fuses_when_conv_lacks_bias(self):
        """Test Conv+BN fusion succeeds when Conv has no bias input."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=False)
        nodes = _build_conv_bn_nodes(3, has_bias=False, is_conv_bn=True)

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_skips_fusion_for_non_depthwise_patterns(self):
        """Test fusion skips non-depthwise Conv (group=1)."""
        # Setup regular (non-depthwise) conv: group=1
        conv_weight = np.random.randn(3, 3, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=True)

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

    def test_fuses_with_nonuniform_scale_and_bias_values(self):
        """Test fusion succeeds with varied BN scale/bias per channel."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        conv_bias = np.ones(3, dtype=np.float32) * 0.5
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        bn_scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        bn_bias = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        bn_mean = np.array([0.0, 0.1, -0.1], dtype=np.float32)
        bn_var = np.array([1.0, 2.0, 0.5], dtype=np.float32)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": create_initializer("bn_scale", bn_scale),
            "bn_bias": create_initializer("bn_bias", bn_bias),
            "bn_mean": create_initializer("bn_mean", bn_mean),
            "bn_var": create_initializer("bn_var", bn_var),
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

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_fuses_with_large_kernel_size(self):
        """Test fusion succeeds with large (5x5) kernel."""
        conv_weight = np.random.randn(16, 1, 5, 5).astype(np.float32)
        initializers = _build_conv_bn_initializers(16, conv_weight, has_bias=True)
        nodes = _build_conv_bn_nodes(16, has_bias=True, is_conv_bn=True)

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Conv"

    def test_empty_node_list_returns_empty(self):
        """Test fusion with empty node list."""
        initializers: dict[str, Any] = {}
        nodes: list[Any] = []

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 0

    def test_passes_through_single_node_graph(self):
        """Test single Conv node returned unchanged from fusion."""
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

    def test_float32_dtype_preserved_after_fusion(self):
        """Test that float32 dtype is preserved in fusion."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=True)
        nodes = _build_conv_bn_nodes(3, has_bias=True, is_conv_bn=True)

        _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers, is_conv_bn=True)

        assert "W" in initializers
        fused_weight = numpy_helper.to_array(initializers["W"])
        assert fused_weight.dtype == np.float32

    def test_applies_correct_fusion_formula_for_order_conv_then_bn(self):
        """Test Conv+BN fusion applies correct weight/bias formulas."""
        # Simple depthwise conv: group=3, all-ones weight, known bias
        conv_weight = np.ones((3, 1, 1, 1), dtype=np.float32)
        conv_bias = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        bn_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        bn_bias_val = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        bn_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bn_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": create_initializer("bn_scale", bn_scale),
            "bn_bias": create_initializer("bn_bias", bn_bias_val),
            "bn_mean": create_initializer("bn_mean", bn_mean),
            "bn_var": create_initializer("bn_var", bn_var),
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

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=True
        )

        assert len(result_nodes) == 1
        fused_weight = numpy_helper.to_array(initializers["W"])
        fused_bias = numpy_helper.to_array(initializers["B"])

        # Formula: new_weight = weight * bn_scale, new_bias = bias * bn_scale + bn_bias
        # new_weight = 1.0 * 2.0 = 2.0
        # new_bias = 2.0 * 2.0 + 1.0 = 5.0
        assert np.allclose(fused_weight, 2.0)
        assert np.allclose(fused_bias, 5.0)

    def test_applies_correct_fusion_formula_for_order_bn_then_conv(self):
        """Test BN+Conv fusion applies correct weight/bias formulas."""
        conv_weight = np.ones((3, 1, 1, 1), dtype=np.float32)
        conv_bias = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        weight_init = create_initializer("W", conv_weight)
        bias_init = create_initializer("B", conv_bias)

        bn_scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        bn_bias_val = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        bn_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        bn_var = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        initializers = {
            "W": weight_init,
            "B": bias_init,
            "bn_scale": create_initializer("bn_scale", bn_scale),
            "bn_bias": create_initializer("bn_bias", bn_bias_val),
            "bn_mean": create_initializer("bn_mean", bn_mean),
            "bn_var": create_initializer("bn_var", bn_var),
        }

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

        assert len(result_nodes) == 1
        fused_weight = numpy_helper.to_array(initializers["W"])
        fused_bias = numpy_helper.to_array(initializers["B"])

        # Formula: new_weight = weight * bn_scale, new_bias = bias + bn_bias
        # new_weight = 1.0 * 2.0 = 2.0
        # new_bias = 2.0 + 1.0 = 3.0
        assert np.allclose(fused_weight, 2.0)
        assert np.allclose(fused_bias, 3.0)

    def test_generates_bias_name_when_conv_lacks_input(self):
        """Test new bias initializer added when Conv has no bias input."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=False)

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

        assert len(result_nodes) == 1
        fused_node = result_nodes[0]
        assert len(fused_node.input) == 3
        bias_name = fused_node.input[2]
        assert bias_name in initializers
        assert "_bias" in bias_name

    @pytest.mark.parametrize("is_conv_bn", [True, False])
    def test_input_output_wiring_preserved_after_fusion(self, is_conv_bn):
        """Test input/output wiring for Conv+BN and BN+Conv."""
        conv_weight = np.random.randn(3, 1, 3, 3).astype(np.float32)
        initializers = _build_conv_bn_initializers(3, conv_weight, has_bias=True)

        if is_conv_bn:
            conv_node = helper.make_node(
                "Conv",
                inputs=["input_X", "W", "B"],
                outputs=["conv_out"],
                group=3,
                name="conv_0",
            )
            bn_node = helper.make_node(
                "BatchNormalization",
                inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["final_output"],
                name="bn_0",
            )
            nodes = [conv_node, bn_node]
        else:
            bn_node = helper.make_node(
                "BatchNormalization",
                inputs=["input_X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["bn_out"],
                name="bn_0",
            )
            conv_node = helper.make_node(
                "Conv",
                inputs=["bn_out", "W", "B"],
                outputs=["final_output"],
                group=3,
                name="conv_0",
            )
            nodes = [bn_node, conv_node]

        result_nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=is_conv_bn
        )

        assert len(result_nodes) == 1
        fused_node = result_nodes[0]
        assert fused_node.input[0] == "input_X"
        assert fused_node.output[0] == "final_output"
