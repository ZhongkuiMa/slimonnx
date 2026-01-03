"""Unit tests for Gemm+BN fusion."""

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGemmBNFusion:
    """Test Gemm+BN fusion."""

    def test_gemm_basic(self):
        """Basic Gemm operation."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

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
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized is not None

    def test_gemm_with_nonzero_bias(self):
        """Gemm with non-zero bias for numerical validation."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Gemm: (1, 3) @ (3, 2) -> (1, 2) with bias
        B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        C = np.array([0.5, 1.5], dtype=np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node("Gemm", inputs=["X", "B", "C"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

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
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)

    def test_bn_gemm_direct_fusion(self):
        """BN → Gemm → fused Gemm (2-node pattern)."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # BN: (1, 3) with 3 channels
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        # Gemm: (1, 3) @ (3, 2) -> (1, 2)
        B = np.eye(3, 2, dtype=np.float32)
        C = np.zeros(2, dtype=np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        gemm_node = helper.make_node("Gemm", inputs=["bn_out", "B", "C"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([bn_node, gemm_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_bn_gemm=True, has_batch_dim=True)

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
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)

    def test_gemm_with_alpha_beta(self):
        """Gemm with alpha=2.0 and beta=0.5 attributes."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Gemm: (1, 3) @ (3, 2) -> (1, 2)
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
            alpha=2.0,
            beta=0.5,
        )

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

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
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)

    def test_gemm_two_input_mode(self):
        """Gemm with 2 inputs (no bias term C)."""
        import onnxruntime as ort

        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Gemm: (1, 3) @ (3, 2) -> (1, 2) - no bias
        B = np.ones((3, 2), dtype=np.float32)

        initializers = [
            create_initializer("B", B),
        ]

        gemm_node = helper.make_node("Gemm", inputs=["X", "B"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)

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
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-4, atol=1e-5)
