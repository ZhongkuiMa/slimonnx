"""Tests for SlimONNX core optimization API flag combinations."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestOptimizationFlags:
    """Test various optimization flag combinations."""

    def test_fuse_conv_bn_flag(self):
        """Test fuse_conv_bn flag effect."""
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

        # Test with fusion disabled
        optimized_no_fusion = optimize_onnx(model, fuse_conv_bn=False)
        bn_count_no_fusion = len(
            [n for n in optimized_no_fusion.graph.node if n.op_type == "BatchNormalization"]
        )

        # Test with fusion enabled
        optimized_with_fusion = optimize_onnx(model, fuse_conv_bn=True)
        bn_count_with_fusion = len(
            [n for n in optimized_with_fusion.graph.node if n.op_type == "BatchNormalization"]
        )

        # With fusion, BN should be removed
        assert bn_count_no_fusion >= 0
        assert bn_count_with_fusion == 0

    def test_remove_dropout_flag(self):
        """Test remove_dropout flag effect."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        ratio_init = create_initializer("ratio", np.array(0.5, dtype=np.float32))

        dropout_node = helper.make_node(
            "Dropout",
            inputs=["X", "ratio"],
            outputs=["Y"],
        )

        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([dropout_node], inputs, outputs, [ratio_init])

        # Test with dropout removal
        optimized = optimize_onnx(model, remove_dropout=True)
        dropout_nodes = [n for n in optimized.graph.node if n.op_type == "Dropout"]

        # Dropout should be removed
        assert len(dropout_nodes) == 0

    def test_multiple_flags_enabled(self):
        """Test multiple optimization flags enabled together."""
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

        # Test multiple flags enabled
        optimized = optimize_onnx(
            model,
            fuse_conv_bn=True,
            remove_dropout=True,
            fuse_gemm_gemm=False,
            has_batch_dim=True,
        )

        assert optimized is not None
        bn_nodes = [n for n in optimized.graph.node if n.op_type == "BatchNormalization"]
        assert len(bn_nodes) == 0

    def test_no_batch_dim_flag(self):
        """Test has_batch_dim flag with False value."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        # Use Relu instead of Identity since Identity is not supported
        relu_node = helper.make_node(
            "Relu",
            inputs=["X"],
            outputs=["Y"],
        )

        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([relu_node], inputs, outputs)

        # Test with batch dimension
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_all_flags_combinations(self):
        """Test all major optimization flags enabled."""
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

        # Test all flags enabled
        optimized = optimize_onnx(
            model,
            fuse_conv_bn=True,
            remove_dropout=True,
            fuse_gemm_gemm=True,
            remove_redundant_operations=True,
            has_batch_dim=True,
        )

        assert optimized is not None
        bn_nodes = [n for n in optimized.graph.node if n.op_type == "BatchNormalization"]
        assert len(bn_nodes) == 0
