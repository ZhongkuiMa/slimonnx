"""Unit tests for depthwise convolution fusion."""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDepthwiseConvBNFusion:
    """Test depthwise Conv+BN → depthwise Conv fusion."""

    def test_executes_standalone_operation(self):
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
        assert optimized

    @pytest.mark.parametrize(
        "conv_first",
        [
            pytest.param(True, id="operation_followed_by_bn"),
            pytest.param(False, id="bn_before_operation"),
        ],
    )
    def test_handles_bn_order_with_depthwise_conv(self, conv_first):
        """Test BN and depthwise Conv handling regardless of node order."""
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

        if conv_first:
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
            node_list = [conv_node, bn_node]
        else:
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
                group=channels,
            )
            node_list = [bn_node, conv_node]

        outputs = [create_tensor_value_info("Y", "float32", [1, channels, 2, 2])]
        model = create_minimal_onnx_model(node_list, inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

    # [REVIEW] Deleted: test_handles_operation_followed_by_bn (merged into test_handles_bn_order_with_depthwise_conv)
    # [REVIEW] Deleted: test_handles_bn_before_operation (merged into test_handles_bn_order_with_depthwise_conv)

    def test_maintains_numerical_equivalence_without_fusion(self):
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

    # [REVIEW] Deleted: test_handles_bn_before_operation (merged into test_handles_bn_order_with_depthwise_conv)

    @pytest.mark.parametrize(
        ("channels", "kernel", "input_size"),
        [
            pytest.param(4, 3, 4, id="multiple_channel_configuration"),
            pytest.param(2, 5, 8, id="larger_kernel_size"),
        ],
    )
    def test_handles_depthwise_conv_configurations(self, channels, kernel, input_size):
        """Test depthwise Conv with various channel/kernel configurations."""
        X = create_tensor_value_info("X", "float32", [1, channels, input_size, input_size])
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

        output_size = input_size - kernel + 1
        outputs = [
            create_tensor_value_info("Y", "float32", [1, channels, output_size, output_size])
        ]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

        # [REVIEW] Deleted: test_handles_multiple_channel_configuration (merged into test_handles_depthwise_conv_configurations)
        # [REVIEW] Deleted: test_handles_larger_kernel_size (merged into test_handles_depthwise_conv_configurations)
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

    # [REVIEW] Deleted: test_handles_larger_kernel_size (merged into test_handles_depthwise_conv_configurations)

    def test_maintains_correctness_with_epsilon_handling(self):
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
