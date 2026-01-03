"""Unit tests for MatMul+Add fusion optimization."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestMatMulAddFusion:
    """Test MatMul+Add → Gemm fusion."""

    def test_basic_fusion_success(self):
        """MatMul(X, W) + b → Gemm with batch=True."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        b = np.array([0.1, 0.2], dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) > 0

    def test_skip_fusion_rank3_input(self):
        """Rank-3 MatMul should NOT fuse to Gemm."""
        X = create_tensor_value_info("X", "float32", [2, 3, 4])
        inputs = [X]

        W = np.random.randn(4, 2).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 3, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "Rank-3 MatMul should NOT fuse"

    def test_no_fusion_without_flag(self):
        """Without fuse_matmul_add=True, fusion should not occur."""
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

        # Optimize WITHOUT fusion flag
        optimized = optimize_onnx(model, fuse_matmul_add=False, has_batch_dim=True)

        # MatMul and Add should still be present
        matmul_nodes = [n for n in optimized.graph.node if n.op_type == "MatMul"]
        assert len(matmul_nodes) >= 1

    def test_fusion_with_bias_variations(self):
        """Bias handling in MatMul+Add fusion - should fuse correctly."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 4])
        inputs = [X]

        W = np.ones((4, 2), dtype=np.float32)
        b = np.array([0.1, 0.2], dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Verify numerical correctness
        test_input = np.ones((1, 4), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_skip_fusion_rank4_input(self):
        """4D tensor (NCHW) should NOT fuse to Gemm."""
        X = create_tensor_value_info("X", "float32", [1, 3, 4, 4])
        inputs = [X]

        # Flatten output would be (1, 48) but input is 4D
        W = np.random.randn(48, 2).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "4D MatMul should NOT fuse to Gemm"

    def test_skip_fusion_multiple_consumers(self):
        """MatMul output used by 2 nodes - fusion should be skipped."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        W = np.ones((3, 2), dtype=np.float32)
        b = np.ones((2,), dtype=np.float32)
        scale = np.ones((2,), dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
            create_initializer("scale", scale),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        # MatMul output used by TWO nodes
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y1"])
        mul_node = helper.make_node("Mul", inputs=["matmul_output", "scale"], outputs=["Y2"])

        outputs = [
            create_tensor_value_info("Y1", "float32", [2, 2]),
            create_tensor_value_info("Y2", "float32", [2, 2]),
        ]
        model = create_minimal_onnx_model(
            [matmul_node, add_node, mul_node], inputs, outputs, initializers
        )

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        # Should not fuse because MatMul output has multiple consumers
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "MatMul with multiple consumers should NOT fuse"

    def test_skip_fusion_missing_weight_initializer(self):
        """Weight is graph input (not constant) - fusion should be skipped."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        W = create_tensor_value_info("W", "float32", [3, 2])
        b_init = np.ones((2,), dtype=np.float32)
        inputs = [X, W]  # W is input, not initializer

        initializers = [create_initializer("b", b_init)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "MatMul with variable weight should NOT fuse"

    def test_skip_fusion_missing_initializer(self):
        """MatMul weight not in initializers - fusion should be skipped."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        W = create_tensor_value_info("W", "float32", [3, 2])
        inputs = [X, W]  # W is input, not initializer

        b = np.array([0.1, 0.2], dtype=np.float32)
        initializers = [create_initializer("b", b)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) == 0, "MatMul+Add with variable weight should NOT fuse"

    def test_numerical_correctness_standard_shapes(self):
        """Standard shape verification - 1x3 @ 3x2 case."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        b = np.array([0.5, 1.5], dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Test with specific input
        test_input = np.ones((1, 3), dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)

    def test_numerical_correctness_varied_values(self):
        """Varied input values verification - identity matrix input."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 5])
        inputs = [X]

        W = np.eye(5, 3, dtype=np.float32)
        b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Test with different input values
        test_input = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)
