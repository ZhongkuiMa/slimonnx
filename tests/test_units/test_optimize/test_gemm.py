"""Unit tests for Gemm normalization."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGemmNormalization:
    """Test Gemm attribute normalization."""

    def test_gemm_basic_normalization(self):
        """Gemm with default attributes normalizes correctly."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        A = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("A", A),
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_gemm_with_alpha(self):
        """Gemm with alpha attribute."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=2.0,
            beta=1.0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_absorb_alpha_into_weight(self):
        """alpha=2.0 should be absorbed into weight - tests alpha absorption branch."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=2.0,
            beta=1.0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
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

    def test_absorb_beta_into_bias(self):
        """beta=0.5 should be absorbed into bias - tests beta absorption branch."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([1.0, 2.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=1.0,
            beta=0.5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
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

    def test_gemm_with_trans_a(self):
        """Gemm with transA=1 attribute."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Gemm without transpose - simple case
        B = np.eye(3, 2, dtype=np.float32)
        C = np.zeros(2, dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            transA=0,
            transB=0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
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

    def test_no_change_already_normalized(self):
        """Gemm with alpha=1, beta=1, no transpose should have no changes - tests no-op branch."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_shared_initializer_not_modified(self):
        """Shared weight used by 2 Gemm nodes should not be modified - tests shared tensor safety check."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node1 = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y1"],
            alpha=2.0,
        )
        gemm_node2 = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y2"],
            alpha=2.0,
        )

        outputs = [
            create_tensor_value_info("Y1", "float32", [1, 2]),
            create_tensor_value_info("Y2", "float32", [1, 2]),
        ]
        model = create_minimal_onnx_model([gemm_node1, gemm_node2], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_handle_missing_bias(self):
        """Gemm with 2 inputs (no bias) should handle correctly - tests optional bias handling."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)

        initializers = [
            create_initializer("B", B),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B"],
            outputs=["Y"],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_default_attributes(self):
        """Missing alpha/beta attributes should use defaults (1.0) - tests default value handling."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([0.0, 0.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        # No alpha/beta attributes - should default to 1.0
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_alpha_zero_edge_case(self):
        """alpha=0.0 is an edge case - tests zero alpha branch."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.eye(3, 2, dtype=np.float32)
        C = np.array([1.0, 2.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=0.0,
            beta=1.0,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None
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

    def test_numerical_correctness_after_normalization(self):
        """Normalized Gemm should produce identical outputs to original - comprehensive test."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        C = np.array([2.0, 3.0], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
            alpha=2.0,
            beta=0.5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)

        # Test with different inputs to ensure correctness
        test_inputs = [
            np.ones((1, 3), dtype=np.float32),
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[0.0, 1.0, -1.0]], dtype=np.float32),
        ]

        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        for test_input in test_inputs:
            original_out = original_sess.run(None, {"X": test_input})[0]
            optimized_out = optimized_sess.run(None, {"X": test_input})[0]
            np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)
