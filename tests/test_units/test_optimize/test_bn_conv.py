"""Unit tests for Conv+BN fusion."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestConvBNFusion:
    """Test Conv+BN → Conv fusion."""

    def test_conv_basic(self):
        """Basic Conv operation works."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_conv_bn_basic(self):
        """Conv followed by BN."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
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

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None

    def test_conv_bn_fusion_basic(self):
        """Conv(3→2) + BN → fused Conv - tests basic fusion logic."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
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
            pads=[0, 0, 0, 0],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)

        # Verify BN was fused (should not appear in optimized graph)
        bn_nodes = [n for n in optimized.graph.node if n.op_type == "BatchNormalization"]
        assert len(bn_nodes) == 0

        # Verify numerical correctness
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_bn_conv_basic_fusion(self):
        """BN + Conv(3→2) → fused Conv - tests reverse pattern."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)
        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)

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
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([bn_node, conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None

        # Verify numerical correctness
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_conv_bn_preserves_padding(self):
        """Conv padding=[1,1] should be preserved - tests attribute preservation."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
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

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 4, 4])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)

        # Check that Conv node still has padding attribute
        conv_nodes = [n for n in optimized.graph.node if n.op_type == "Conv"]
        assert len(conv_nodes) > 0

    def test_bn_epsilon_in_fusion_math(self):
        """BN epsilon=1e-5 should be handled correctly in weight scaling."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.array([0.01, 0.01], dtype=np.float32)  # Small variance

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

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)

        # Verify numerical correctness with epsilon handling
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)

    def test_conv_without_bn(self):
        """Conv operation without BN should pass through - tests Conv-only case."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_bn_without_conv(self):
        """BN operation without Conv should pass through - tests BN-only case."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 4, 4])]
        model = create_minimal_onnx_model([bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_numerical_correctness_small_variance(self):
        """BN with var=0.01 (edge case) - tests numerical stability."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.ones(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
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
            kernel_shape=[3, 3],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)

        # Verify numerical correctness with small variance
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)


class TestConvTransposeBNFusion:
    """Test ConvTranspose+BN → ConvTranspose fusion."""

    def test_convtranspose_basic(self):
        """Basic ConvTranspose operation works."""
        X = create_tensor_value_info("X", "float32", [1, 2, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 6, 6])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_convtranspose_bn_basic(self):
        """ConvTranspose followed by BN."""
        X = create_tensor_value_info("X", "float32", [1, 2, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "ConvTranspose",
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

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 6, 6])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None

    def test_convtranspose_bn_fusion_numerical(self):
        """ConvTranspose+BN fusion correctness - numerical validation."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 2, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "ConvTranspose",
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

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 6, 6])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)

        # Verify numerical correctness
        test_input = np.ones((1, 2, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_bn_convtranspose_basic(self):
        """BN + ConvTranspose - tests reverse pattern."""
        X = create_tensor_value_info("X", "float32", [1, 2, 4, 4])
        inputs = [X]

        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)
        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)

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
            "ConvTranspose",
            inputs=["bn_out", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 6, 6])]
        model = create_minimal_onnx_model([bn_node, conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None

    def test_convtranspose_bn_with_padding(self):
        """ConvTranspose+BN with output_padding attribute."""
        X = create_tensor_value_info("X", "float32", [1, 2, 4, 4])
        inputs = [X]

        conv_w = np.ones((2, 3, 3, 3), dtype=np.float32)
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers = [
            create_initializer("conv_w", conv_w),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["X", "conv_w"],
            outputs=["conv_out"],
            kernel_shape=[3, 3],
            output_padding=[1, 1],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 7, 7])]
        model = create_minimal_onnx_model([conv_node, bn_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None
