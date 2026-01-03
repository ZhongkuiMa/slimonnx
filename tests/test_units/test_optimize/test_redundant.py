"""Unit tests for redundant operation removal."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestRedundantOperations:
    """Test optimization on models with redundant patterns."""

    def test_optimize_simple_add(self):
        """Optimize model with Add operation."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        b = np.ones((1, 3), dtype=np.float32)
        initializers = [create_initializer("b", b)]

        add_node = helper.make_node("Add", inputs=["X", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_optimize_mul_operation(self):
        """Optimize model with Mul operation."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        scale = 2.0 * np.ones((1, 3), dtype=np.float32)
        initializers = [create_initializer("scale", scale)]

        mul_node = helper.make_node("Mul", inputs=["X", "scale"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([mul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_optimize_chained_operations(self):
        """Optimize model with chained operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        b1 = np.ones((1, 3), dtype=np.float32)
        b2 = 2.0 * np.ones((1, 3), dtype=np.float32)
        initializers = [
            create_initializer("b1", b1),
            create_initializer("b2", b2),
        ]

        add_node = helper.make_node("Add", inputs=["X", "b1"], outputs=["temp"])
        mul_node = helper.make_node("Mul", inputs=["temp", "b2"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node, mul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_matmul_then_add_fuses(self):
        """MatMul followed by Add should fuse into Gemm - tests fusion across operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.eye(3, 2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["temp"])
        add_node = helper.make_node("Add", inputs=["temp", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        # MatMul+Add fusion produces Gemm
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) > 0

    def test_transpose_operations_handled(self):
        """Transpose operations should be handled correctly - tests transpose logic."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.eye(2, 3, dtype=np.float32).T
        initializers = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_add_nonzero_value_kept(self):
        """X + non_zero value should be kept - tests non-zero threshold."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Small but non-zero value
        val = np.array([0.001, 0.001, 0.001], dtype=np.float32)
        initializers = [create_initializer("val", val)]

        add_node = helper.make_node("Add", inputs=["X", "val"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
        # Non-zero Add should be kept
        add_nodes = [n for n in optimized.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 1

    def test_add_zero_preserved(self):
        """X + 0.0 preservation - verify optimization doesn't crash."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        zero = np.zeros((1, 3), dtype=np.float32)
        initializers = [create_initializer("zero", zero)]

        add_node = helper.make_node("Add", inputs=["X", "zero"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness
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

    def test_sub_zero_preserved(self):
        """X - 0.0 preservation - verify optimization doesn't crash."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        zero = np.zeros((1, 3), dtype=np.float32)
        initializers = [create_initializer("zero", zero)]

        sub_node = helper.make_node("Sub", inputs=["X", "zero"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([sub_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness
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

    def test_mul_one_preserved(self):
        """X * 1.0 preservation - verify optimization doesn't crash."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        one = np.ones((1, 3), dtype=np.float32)
        initializers = [create_initializer("one", one)]

        mul_node = helper.make_node("Mul", inputs=["X", "one"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([mul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness
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

    def test_div_one_preserved(self):
        """X / 1.0 preservation - verify optimization doesn't crash."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        one = np.ones((1, 3), dtype=np.float32)
        initializers = [create_initializer("one", one)]

        div_node = helper.make_node("Div", inputs=["X", "one"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([div_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        # Verify numerical correctness
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

    def test_add_nonzero_kept(self):
        """X + 0.001 should be kept - tests non-zero threshold."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        small_val = 0.001 * np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("val", small_val)]

        add_node = helper.make_node("Add", inputs=["X", "val"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        # Add with non-zero should be kept
        add_nodes = [n for n in optimized.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 1

    def test_graph_optimization_multi_output(self):
        """Multiple outputs should be handled correctly - tests multi-output logic."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W1 = np.ones((3, 2), dtype=np.float32)
        W2 = np.ones((3, 2), dtype=np.float32)
        initializers = [
            create_initializer("W1", W1),
            create_initializer("W2", W2),
        ]

        matmul1 = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["Y1"])
        matmul2 = helper.make_node("MatMul", inputs=["X", "W2"], outputs=["Y2"])

        outputs = [
            create_tensor_value_info("Y1", "float32", [1, 2]),
            create_tensor_value_info("Y2", "float32", [1, 2]),
        ]
        model = create_minimal_onnx_model([matmul1, matmul2], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_numerical_correctness_after_chained_ops(self):
        """Graph outputs should be unchanged after chained operation optimization."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Create model with chained ops: MatMul → Add
        W = np.eye(3, 2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Run both and compare
        test_input = np.ones((1, 3), dtype=np.float32)

        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = original_sess.run(None, {"X": test_input})[0]

        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_out = optimized_sess.run(None, {"X": test_input})[0]

        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)
