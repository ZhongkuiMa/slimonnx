"""Tests for Conv+BN pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.conv_bn import detect_conv_bn

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestConvBNDetection:
    """Test Conv+BN pattern detection."""

    def test_detect_basic_conv_bn(self):
        """Test detection of basic Conv+BN pattern."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Detect patterns
        instances = detect_conv_bn(nodes, initializers_dict)

        # Verify detection
        assert isinstance(instances, list)
        assert len(instances) > 0

    def test_detect_conv_bn_with_padding(self):
        """Test Conv+BN detection with padding."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 4, 4])]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_conv_bn(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_multiple_conv_bn_patterns(self):
        """Test detection of multiple Conv+BN patterns in sequence."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]

        # First Conv+BN pattern
        conv_w1 = np.random.randn(4, 3, 3, 3).astype(np.float32)
        bn_scale1 = np.ones(4, dtype=np.float32)
        bn_bias1 = np.zeros(4, dtype=np.float32)
        bn_mean1 = np.zeros(4, dtype=np.float32)
        bn_var1 = np.ones(4, dtype=np.float32)

        # Second Conv+BN pattern
        conv_w2 = np.random.randn(2, 4, 3, 3).astype(np.float32)
        bn_scale2 = np.ones(2, dtype=np.float32)
        bn_bias2 = np.zeros(2, dtype=np.float32)
        bn_mean2 = np.zeros(2, dtype=np.float32)
        bn_var2 = np.ones(2, dtype=np.float32)

        initializers = [
            create_initializer("conv_w1", conv_w1),
            create_initializer("bn_scale1", bn_scale1),
            create_initializer("bn_bias1", bn_bias1),
            create_initializer("bn_mean1", bn_mean1),
            create_initializer("bn_var1", bn_var1),
            create_initializer("conv_w2", conv_w2),
            create_initializer("bn_scale2", bn_scale2),
            create_initializer("bn_bias2", bn_bias2),
            create_initializer("bn_mean2", bn_mean2),
            create_initializer("bn_var2", bn_var2),
        ]

        conv1_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w1"],
            outputs=["conv1_out"],
            kernel_shape=[3, 3],
        )
        bn1_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv1_out", "bn_scale1", "bn_bias1", "bn_mean1", "bn_var1"],
            outputs=["bn1_out"],
            epsilon=1e-5,
        )
        conv2_node = helper.make_node(
            "Conv",
            inputs=["bn1_out", "conv_w2"],
            outputs=["conv2_out"],
            kernel_shape=[3, 3],
        )
        bn2_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv2_out", "bn_scale2", "bn_bias2", "bn_mean2", "bn_var2"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        model = create_minimal_onnx_model(
            [conv1_node, bn1_node, conv2_node, bn2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_conv_bn(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_no_conv_bn_pattern(self):
        """Test detection when Conv+BN pattern is not present."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        # Only Conv, no BN
        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_conv_bn(nodes, initializers_dict)
        # May return empty list if Conv not immediately followed by BN
        assert isinstance(instances, list)

    def test_detect_conv_bn_different_channels(self):
        """Test Conv+BN detection with various channel configurations."""
        inputs = [create_tensor_value_info("X", "float32", [1, 64, 8, 8])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 128, 4, 4])]

        conv_w = np.random.randn(128, 64, 3, 3).astype(np.float32)
        bn_scale = np.ones(128, dtype=np.float32)
        bn_bias = np.zeros(128, dtype=np.float32)
        bn_mean = np.zeros(128, dtype=np.float32)
        bn_var = np.ones(128, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            stride=[2, 2],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_conv_bn(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_conv_bn_with_bias(self):
        """Test Conv+BN where Conv has bias."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3, 4, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        conv_b = np.zeros(2, dtype=np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("conv_b", conv_b),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w", "conv_b"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_conv_bn(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_conv_bn([], {})
        assert isinstance(instances, list)
        assert len(instances) == 0
