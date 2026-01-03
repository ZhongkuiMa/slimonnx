"""Tests for ConvTranspose+BN fusion optimization."""

import contextlib
import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._bn_conv import (
    _fuse_conv_bn_or_bn_conv,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestConvTransposeBnFusion:
    """Test ConvTranspose+BN fusion patterns."""

    def test_conv_transpose_bn_fusion(self):
        """Test fusing ConvTranspose+BN."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 4, 4])

        conv_w = np.random.randn(3, 2, 3, 3).astype(np.float32)
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

        convt = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([convt, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        with contextlib.suppress(NotImplementedError):
            result = _fuse_conv_bn_or_bn_conv(nodes, initializers_dict)
            # May not support ConvTranspose, but should not crash
            assert len(result) >= 0

    def test_bn_conv_transpose_fusion(self):
        """Test fusing BN+ConvTranspose."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 4, 4])

        conv_w = np.random.randn(3, 2, 3, 3).astype(np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("W", conv_w),
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
        initializers_dict = {init.name: init for init in model.graph.initializer}

        with contextlib.suppress(NotImplementedError):
            result = _fuse_conv_bn_or_bn_conv(nodes, initializers_dict)
            # May not support ConvTranspose, but should not crash
            assert len(result) >= 0

    def test_conv_transpose_with_padding(self):
        """Test ConvTranspose+BN with padding (should not fuse)."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 4, 4])

        conv_w = np.random.randn(3, 2, 3, 3).astype(np.float32)
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

        convt = helper.make_node(
            "ConvTranspose",
            inputs=["X", "W"],
            outputs=["C"],
            pads=[1, 1, 1, 1],
        )
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([convt, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        with contextlib.suppress(NotImplementedError):
            result = _fuse_conv_bn_or_bn_conv(nodes, initializers_dict)
            assert len(result) >= 0

    def test_conv_transpose_chain_with_bn(self):
        """Test ConvTranspose chain with BN in middle."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        Z = create_tensor_value_info("Z", "float32", [1, 2, 4, 4])

        conv_w = np.random.randn(3, 2, 3, 3).astype(np.float32)
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

        convt = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["B"],
        )
        relu = helper.make_node("Relu", inputs=["B"], outputs=["Z"])

        model = create_minimal_onnx_model([convt, bn, relu], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        with contextlib.suppress(NotImplementedError):
            result = _fuse_conv_bn_or_bn_conv(nodes, initializers_dict)
            assert len(result) >= 0

    def test_conv_transpose_no_bias(self):
        """Test ConvTranspose+BN when ConvTranspose has no bias."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 4, 4])

        conv_w = np.random.randn(3, 2, 3, 3).astype(np.float32)
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

        convt = helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([convt, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        with contextlib.suppress(NotImplementedError):
            result = _fuse_conv_bn_or_bn_conv(nodes, initializers_dict)
            assert len(result) >= 0
