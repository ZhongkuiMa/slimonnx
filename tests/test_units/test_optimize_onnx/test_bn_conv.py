"""Unit tests for Conv+BN fusion."""

__docformat__ = "restructuredtext"

import numpy as np
import onnxruntime as ort
import pytest
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx


class TestConvBNFusion:
    """Test Conv+BN -> Conv fusion."""

    @pytest.mark.parametrize(
        "weight_init",
        [
            pytest.param(
                lambda shape: np.random.randn(*shape).astype(np.float32),
                id="random_weights",
            ),
            pytest.param(
                lambda shape: np.ones(shape, dtype=np.float32),
                id="ones_weights",
            ),
        ],
    )
    def test_passes_through_without_batchnorm(self, weight_init):
        """Conv without BN should pass through unchanged for both random and constant weights."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = weight_init((2, 3, 3, 3))
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
        assert optimized

    def test_fuses_into_single_operation(self):
        """Conv followed by BN should fuse into single Conv."""
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
        assert optimized

    def test_produces_output_matching_unfused_computation(self):
        """Conv+BN fusion must produce numerically equivalent output."""
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
        original_out = np.asarray(original_sess.run(None, {"X": test_input})[0])
        optimized_out = np.asarray(optimized_sess.run(None, {"X": test_input})[0])
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_fuses_when_bn_precedes_operation(self):
        """BN followed by Conv should fuse into single Conv."""
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
        assert optimized

        # Verify numerical correctness
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = np.asarray(original_sess.run(None, {"X": test_input})[0])
        optimized_out = np.asarray(optimized_sess.run(None, {"X": test_input})[0])
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_preserves_padding_attribute_after_fusion(self):
        """Conv+BN fusion should preserve Conv padding attribute."""
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

    @pytest.mark.parametrize(
        "bn_bias",
        [
            pytest.param(np.zeros(2, dtype=np.float32), id="zero_bias"),
            pytest.param(np.ones(2, dtype=np.float32), id="ones_bias"),
        ],
    )
    def test_fusion_numerical_correctness_with_small_variance(self, bn_bias):
        """Conv+BN fusion produces numerically correct output with small BN variance."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        bn_scale = np.ones(2, dtype=np.float32)
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

        # Verify numerical correctness with small variance
        test_input = np.ones((1, 3, 4, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = np.asarray(original_sess.run(None, {"X": test_input})[0])
        optimized_out = np.asarray(optimized_sess.run(None, {"X": test_input})[0])
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)

    def test_passes_through_standalone_batchnorm(self):
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
        assert optimized


class TestConvTransposeBNFusion:
    """Test ConvTranspose+BN -> ConvTranspose fusion."""

    def test_passes_through_without_batchnorm_transpose(self):
        """ConvTranspose without BN should pass through unchanged."""
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
        assert optimized

    def test_fuses_operation_then_bn_into_single_transpose(self):
        """ConvTranspose followed by BN should fuse into single ConvTranspose."""
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
        assert optimized

    def test_produces_numerically_equivalent_output_after_fusing(self):
        """ConvTranspose+BN fusion produces output matching un-fused computation."""
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
        original_out = np.asarray(original_sess.run(None, {"X": test_input})[0])
        optimized_out = np.asarray(optimized_sess.run(None, {"X": test_input})[0])
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_fuses_when_bn_precedes_operation_transpose(self):
        """BN followed by ConvTranspose should fuse into single ConvTranspose."""
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
        assert optimized

    def test_preserves_output_padding_attribute_after_fusion_transpose(self):
        """ConvTranspose+BN fusion should preserve output_padding attribute."""
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
        assert optimized
