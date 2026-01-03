"""Unit tests for depthwise convolution fusion."""

import numpy as np
import onnxruntime as ort
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDepthwiseConvBNFusion:
    """Test depthwise Conv+BN → depthwise Conv fusion."""

    def test_depthwise_conv_basic(self):
        """Basic depthwise Conv operation."""
        # Depthwise: out_channels = in_channels, group = in_channels
        channels = 2
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        # Depthwise weight shape: (out_channels, 1, kernel, kernel)
        # out_channels = in_channels for depthwise, so (2, 1, 3, 3)
        conv_w = np.random.randn(channels, 1, kernel, kernel).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[kernel, kernel],
            group=channels,  # Depthwise: group == in_channels
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_depthwise_conv_bn_basic(self):
        """Depthwise Conv followed by BN - optimizer handles without fusion."""
        channels = 2
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        conv_w = np.ones((channels, 1, kernel, kernel), dtype=np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,  # Depthwise
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Note: depthwise fusion (group > 1) is not supported, so skip fusion flag
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_depthwise_conv_bn_fusion_numerical(self):
        """Depthwise Conv+BN handling - optimizer processes without fusion."""
        channels = 3
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        conv_w = np.ones((channels, 1, kernel, kernel), dtype=np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,  # Depthwise
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Depthwise fusion (group > 1) is not supported
        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness
        test_input = np.ones((1, channels, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_bn_depthwise_conv_basic(self):
        """BN + depthwise Conv - reverse pattern handled without fusion."""
        channels = 2
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)
        conv_w = np.ones((channels, 1, kernel, kernel), dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
            create_initializer("conv_w", conv_w),
        ]

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        conv_node = helper.make_node(
            "Conv",
            inputs=["bn_out", "conv_w"],
            outputs=["Y"],
            kernel_shape=[kernel, kernel],
            group=channels,  # Depthwise
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([bn_node, conv_node], inputs, outputs, initializers)

        # Depthwise fusion not supported
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_depthwise_conv_multiple_channels(self):
        """Depthwise Conv with multiple channels (4 channels) - optimizer handles gracefully."""
        channels = 4
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(channels, 1, kernel, kernel).astype(np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Depthwise fusion not supported
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_depthwise_conv_with_padding(self):
        """Depthwise Conv with padding preservation."""
        channels = 2
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        conv_w = np.ones((channels, 1, kernel, kernel), dtype=np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,
            pads=[1, 1, 1, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 4, 4])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Depthwise fusion not supported
        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify Conv node exists
        conv_nodes = [n for n in optimized.graph.node if n.op_type == "Conv"]
        assert len(conv_nodes) > 0

    def test_depthwise_conv_large_kernel(self):
        """Depthwise Conv with larger kernel size (5x5)."""
        channels = 2
        kernel = 5

        X = create_tensor_value_info("X", "float32", [1, channels, 8, 8])
        inputs = [X]

        conv_w = np.random.randn(channels, 1, kernel, kernel).astype(np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.ones(channels, dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 4, 4])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Depthwise fusion not supported
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_depthwise_conv_bn_epsilon(self):
        """Depthwise Conv+BN with epsilon handling."""
        channels = 2
        kernel = 3

        X = create_tensor_value_info("X", "float32", [1, channels, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(channels, 1, kernel, kernel).astype(np.float32)
        bn_scale = np.ones(channels, dtype=np.float32)
        bn_bias = np.zeros(channels, dtype=np.float32)
        bn_mean = np.zeros(channels, dtype=np.float32)
        bn_var = np.array([0.01, 0.01], dtype=np.float32)

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
            kernel_shape=[kernel, kernel],
            group=channels,
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        # Depthwise fusion not supported
        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness with epsilon
        test_input = np.ones((1, channels, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)
