"""Unit tests for error handling in optimization functions."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestConvErrorHandling:
    """Test error handling in Conv optimization."""

    def test_conv_with_group_greater_than_one(self):
        """Conv with group > 1 (non-depthwise grouped conv) should handle gracefully."""
        X = create_tensor_value_info("X", "float32", [1, 6, 4, 4])
        inputs = [X]

        # Group=2: split 6 input channels into 2 groups of 3 channels each
        # Output: 4 output channels (2 per group)
        conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            group=2,  # Non-depthwise group convolution
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 4, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        # Should handle without raising exception
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_conv_without_weight(self):
        """Conv without weight initializer - should handle gracefully."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        # Note: weight is not provided as initializer
        inputs = [X, create_tensor_value_info("W", "float32", [2, 3, 3, 3])]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "W"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs)

        # Should handle gracefully
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_bn_missing_parameters(self):
        """BN with missing optional parameters."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        # Minimal BN: only scale and bias
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
        ]

        # BN with only scale and bias (no mean/var)
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias"],
            outputs=["Y"],
            epsilon=1e-5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 3, 4, 4])]
        model = create_minimal_onnx_model([bn_node], inputs, outputs, initializers)

        # Should handle gracefully
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None


class TestGemmErrorHandling:
    """Test error handling in Gemm optimization."""

    def test_gemm_with_trans_a(self):
        """Gemm with transA=1 - should handle gracefully."""
        A = create_tensor_value_info("A", "float32", [3, 2])
        inputs = [A]

        B_data = np.random.randn(3, 2).astype(np.float32)
        initializers = [create_initializer("B", B_data)]

        # transA=1 means A is transposed
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["A", "B"],
            outputs=["Y"],
            transA=1,  # A is transposed
            transB=0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        # Should handle gracefully
        optimized = optimize_onnx(model, has_batch_dim=False)
        assert optimized is not None

    def test_gemm_with_alpha_beta(self):
        """Gemm with non-default alpha/beta parameters."""
        A = create_tensor_value_info("A", "float32", [2, 3])
        inputs = [A]

        B_data = np.random.randn(3, 2).astype(np.float32)
        C_data = np.random.randn(2).astype(np.float32)
        initializers = [
            create_initializer("B", B_data),
            create_initializer("C", C_data),
        ]

        # Non-default alpha/beta values
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["A", "B", "C"],
            outputs=["Y"],
            alpha=2.0,
            beta=0.5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        # Should handle gracefully
        optimized = optimize_onnx(model, has_batch_dim=False)
        assert optimized is not None


class TestInvalidShapeHandling:
    """Test error handling for invalid tensor shapes."""

    def test_matmul_incompatible_shapes(self):
        """MatMul with incompatible shapes - operation may fail at runtime."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        # Weight shape incompatible with X: (2, 5) but X is [2, 3]
        W_data = np.random.randn(2, 5).astype(np.float32)
        initializers = [create_initializer("W", W_data)]

        matmul_node = helper.make_node(
            "MatMul",
            inputs=["X", "W"],
            outputs=["Y"],
        )

        # Output shape will be undefined at graph construction
        outputs = [create_tensor_value_info("Y", "float32", [2, 5])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        # Optimizer should handle without crashing
        optimized = optimize_onnx(model, has_batch_dim=False)
        assert optimized is not None

    def test_empty_graph(self):
        """Empty model with no nodes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        # Identity node to make graph valid
        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([identity], [X], [Y])

        # Remove identity to create empty graph scenario
        model.graph.ClearField("node")

        # Should handle empty graph gracefully
        optimized = optimize_onnx(model, has_batch_dim=False)
        assert optimized is not None


class TestOptimizationFlagCombinations:
    """Test error handling for various optimization flag combinations."""

    def test_all_flags_enabled(self):
        """Test with all optimization flags enabled."""
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

        # Test with multiple flags enabled
        optimized = optimize_onnx(
            model,
            fuse_conv_bn=True,
            remove_dropout=True,
            fuse_gemm_gemm=True,
            has_batch_dim=True,
        )
        assert optimized is not None

    def test_conflicting_flag_defaults(self):
        """Test behavior when using default flag values."""
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

        # Should handle default flags gracefully
        optimized = optimize_onnx(model)
        assert optimized is not None


class TestEdgeCaseShapes:
    """Test optimization with edge case tensor shapes."""

    def test_single_pixel_conv(self):
        """Conv with 1x1 kernel on small input."""
        X = create_tensor_value_info("X", "float32", [1, 3, 2, 2])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 1, 1).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[1, 1],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_large_batch_size(self):
        """Conv with large batch size."""
        batch_size = 1024
        X = create_tensor_value_info("X", "float32", [batch_size, 3, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(2, 3, 3, 3).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[3, 3],
        )

        outputs = [create_tensor_value_info("Y", "float32", [batch_size, 2, 2, 2])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_many_channels(self):
        """Conv with many input/output channels."""
        X = create_tensor_value_info("X", "float32", [1, 512, 4, 4])
        inputs = [X]

        conv_w = np.random.randn(256, 512, 1, 1).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[1, 1],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 256, 4, 4])]
        model = create_minimal_onnx_model([conv_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
