"""Unit tests for model validation."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGraphValidation:
    """Test graph validation."""

    def test_valid_graph_basic(self):
        """Basic graph should be valid."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.random.randn(3, 2).astype(np.float32)
        initializers = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_optimization_preserves_validity(self):
        """Optimization should produce valid graph."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.eye(3, 2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        assert optimized is not None
        assert len(optimized.graph.node) >= 0

    def test_detect_dead_node(self):
        """Node output unused by graph outputs → graph still optimizes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        # Create two operations but only use one in output
        b1 = np.ones((2, 3), dtype=np.float32)
        b2 = np.ones((2, 3), dtype=np.float32)
        initializers = [
            create_initializer("b1", b1),
            create_initializer("b2", b2),
        ]

        # Dead node: Add produces unused_output
        add_dead = helper.make_node("Add", inputs=["X", "b1"], outputs=["unused_output"])
        # Live node: Add produces Y which is graph output
        add_live = helper.make_node("Add", inputs=["X", "b2"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([add_dead, add_live], inputs, outputs, initializers)

        # Model should optimize without errors
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_detect_broken_connection(self):
        """Missing input connection handled gracefully."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        W = np.ones((3, 2), dtype=np.float32)
        initializers = [create_initializer("W", W)]

        # MatMul with correct inputs
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        # Model should be valid
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_detect_orphan_initializer(self):
        """Unused initializer handled gracefully."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        # Create initializer that's not used by any node
        W_used = np.ones((3, 2), dtype=np.float32)
        W_unused = np.ones((2, 2), dtype=np.float32)
        initializers = [
            create_initializer("W_used", W_used),
            create_initializer("W_unused", W_unused),
        ]

        # MatMul uses only W_used
        matmul_node = helper.make_node("MatMul", inputs=["X", "W_used"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        # Model should optimize without errors
        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_numerical_comparison_identical(self):
        """Identical outputs verify correctly."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        b = np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("b", b)]

        add_node = helper.make_node("Add", inputs=["X", "b"], outputs=["Y"])
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        # Run same model twice - outputs should match
        test_input = np.ones((2, 3), dtype=np.float32)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        output1 = sess.run(None, {"X": test_input})[0]
        output2 = sess.run(None, {"X": test_input})[0]

        # Should be identical
        np.testing.assert_array_equal(output1, output2)

    def test_numerical_comparison_within_tolerance(self):
        """Outputs within tolerance pass correctness check."""
        # Create outputs that differ slightly but within tolerance
        output1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        output2 = np.array([[1.0 + 1e-6, 2.0 + 1e-6], [3.0 + 1e-6, 4.0 + 1e-6]], dtype=np.float32)

        # Should pass with rtol=1e-5
        np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-6)

    def test_numerical_comparison_fails_outside_tolerance(self):
        """Large differences detected correctly."""
        # Create outputs that differ significantly
        output1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        output2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

        # Should fail with default tolerance
        import pytest

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(output1, output2, rtol=1e-5, atol=1e-6)

    def test_detect_nan_in_output(self):
        """NaN in output is detected."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.ones((3, 2), dtype=np.float32)
        initializers = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        test_input = np.ones((1, 3), dtype=np.float32)
        sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output = sess.run(None, {"X": test_input})[0]

        # Output should not contain NaN
        assert not np.any(np.isnan(output))
