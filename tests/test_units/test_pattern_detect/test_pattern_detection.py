"""Unit tests for pattern detection API."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestPatternDetection:
    """Test pattern detection in models."""

    def test_detect_matmul_add_pattern(self):
        """Detect MatMul+Add pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.random.randn(3, 2).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        # Optimize to detect pattern
        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        assert optimized is not None

    def test_empty_graph(self):
        """Empty graph should not error."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]
        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]

        model = create_minimal_onnx_model([], inputs, outputs, [])
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_detect_conv_bn_pattern(self):
        """Conv→BN pattern detected."""
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

        # Optimize to detect and handle pattern
        optimized = optimize_onnx(model, fuse_conv_bn=True, has_batch_dim=True)
        assert optimized is not None

    def test_detect_gemm_reshape_bn_pattern(self):
        """Gemm→Reshape→BN pattern detected."""
        X = create_tensor_value_info("X", "float32", [1, 6])
        inputs = [X]

        B = np.ones((6, 4), dtype=np.float32)
        C = np.zeros(4, dtype=np.float32)
        shape = np.array([1, 2, 2], dtype=np.int64)
        bn_scale = np.ones(1, dtype=np.float32)
        bn_bias = np.zeros(1, dtype=np.float32)
        bn_mean = np.zeros(1, dtype=np.float32)
        bn_var = np.ones(1, dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
            create_initializer("shape", shape),
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        gemm_node = helper.make_node("Gemm", inputs=["X", "B", "C"], outputs=["gemm_out"])
        reshape_node = helper.make_node(
            "Reshape", inputs=["gemm_out", "shape"], outputs=["reshape_out"]
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["reshape_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 2])]
        model = create_minimal_onnx_model(
            [gemm_node, reshape_node, bn_node], inputs, outputs, initializers
        )

        optimized = optimize_onnx(model, fuse_gemm_reshape_bn=True, has_batch_dim=True)
        assert optimized is not None

    def test_detect_multiple_patterns_in_graph(self):
        """Graph with 3 different patterns."""
        # Pattern 1: MatMul+Add
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W1 = np.ones((3, 2), dtype=np.float32)
        b = np.ones(2, dtype=np.float32)

        W2 = np.ones((2, 2), dtype=np.float32)

        initializers = [
            create_initializer("W1", W1),
            create_initializer("b", b),
            create_initializer("W2", W2),
        ]

        # Pattern 1: MatMul+Add
        matmul_node = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["temp1"])

        # Pattern 2: MatMul (single)
        matmul2_node = helper.make_node("MatMul", inputs=["temp1", "W2"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model(
            [matmul_node, add_node, matmul2_node], inputs, outputs, initializers
        )

        # Should handle multiple patterns
        optimized = optimize_onnx(
            model, fuse_matmul_add=True, constant_folding=True, has_batch_dim=True
        )
        assert optimized is not None

    def test_no_false_positives_similar_pattern(self):
        """MatMul→ReLU NOT detected as MatMul→Add."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.ones((3, 2), dtype=np.float32)
        initializers = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        # ReLU instead of Add
        relu_node = helper.make_node("Relu", inputs=["matmul_out"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, relu_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Should NOT create Gemm (pattern doesn't match)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "MatMul→ReLU should NOT fuse to Gemm"

        # MatMul and ReLU should both still be present
        matmul_nodes = [n for n in optimized.graph.node if n.op_type == "MatMul"]
        relu_nodes = [n for n in optimized.graph.node if n.op_type == "Relu"]
        assert len(matmul_nodes) >= 1
        assert len(relu_nodes) >= 1
