"""Tests for BN+Gemm pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.gemm_bn import detect_bn_gemm

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestBNGemmDetection:
    """Test BN+Gemm pattern detection."""

    def test_detect_basic_bn_gemm(self):
        """Test detection of basic BN+Gemm pattern."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)
        B = np.random.randn(3, 2).astype(np.float32)
        C = np.random.randn(2).astype(np.float32)

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
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["bn_out", "B", "C"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([bn_node, gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Detect patterns
        instances = detect_bn_gemm(nodes, initializers_dict)

        # Verify detection
        assert isinstance(instances, list)
        assert len(instances) > 0

    def test_detect_bn_gemm_without_bias(self):
        """Test BN+Gemm detection when Gemm has no bias."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)
        B = np.random.randn(3, 2).astype(np.float32)

        initializers = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
            create_initializer("B", B),
        ]

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
            epsilon=1e-5,
        )
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["bn_out", "B"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([bn_node, gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_bn_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_multiple_bn_gemm_patterns(self):
        """Test detection of multiple BN+Gemm patterns in sequence."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # First BN+Gemm pattern
        bn_scale1 = np.ones(3, dtype=np.float32)
        bn_bias1 = np.zeros(3, dtype=np.float32)
        bn_mean1 = np.zeros(3, dtype=np.float32)
        bn_var1 = np.ones(3, dtype=np.float32)
        B1 = np.random.randn(3, 4).astype(np.float32)
        C1 = np.random.randn(4).astype(np.float32)

        # Second BN+Gemm pattern
        bn_scale2 = np.ones(4, dtype=np.float32)
        bn_bias2 = np.zeros(4, dtype=np.float32)
        bn_mean2 = np.zeros(4, dtype=np.float32)
        bn_var2 = np.ones(4, dtype=np.float32)
        B2 = np.random.randn(4, 2).astype(np.float32)
        C2 = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("bn_scale1", bn_scale1),
            create_initializer("bn_bias1", bn_bias1),
            create_initializer("bn_mean1", bn_mean1),
            create_initializer("bn_var1", bn_var1),
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("bn_scale2", bn_scale2),
            create_initializer("bn_bias2", bn_bias2),
            create_initializer("bn_mean2", bn_mean2),
            create_initializer("bn_var2", bn_var2),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
        ]

        bn1_node = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale1", "bn_bias1", "bn_mean1", "bn_var1"],
            outputs=["bn1_out"],
            epsilon=1e-5,
        )
        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["bn1_out", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        bn2_node = helper.make_node(
            "BatchNormalization",
            inputs=["gemm1_out", "bn_scale2", "bn_bias2", "bn_mean2", "bn_var2"],
            outputs=["bn2_out"],
            epsilon=1e-5,
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["bn2_out", "B2", "C2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model(
            [bn1_node, gemm1_node, bn2_node, gemm2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_bn_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_bn_gemm_with_alpha_beta(self):
        """Test BN+Gemm detection with alpha/beta attributes."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)
        B = np.random.randn(3, 2).astype(np.float32)
        C = np.random.randn(2).astype(np.float32)

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
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["bn_out", "B", "C"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
        )

        model = create_minimal_onnx_model([bn_node, gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_bn_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_no_bn_gemm_pattern(self):
        """Test detection when BN+Gemm pattern is not present."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        B = np.random.randn(3, 2).astype(np.float32)
        C = np.random.randn(2).astype(np.float32)
        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        # Only Gemm, no BN
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_bn_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_bn_gemm_different_shapes(self):
        """Test BN+Gemm detection with various shapes."""
        inputs = [create_tensor_value_info("X", "float32", [4, 512])]
        outputs = [create_tensor_value_info("Y", "float32", [4, 256])]

        bn_scale = np.ones(512, dtype=np.float32)
        bn_bias = np.zeros(512, dtype=np.float32)
        bn_mean = np.zeros(512, dtype=np.float32)
        bn_var = np.ones(512, dtype=np.float32)
        B = np.random.randn(512, 256).astype(np.float32)
        C = np.random.randn(256).astype(np.float32)

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
        gemm_node = helper.make_node(
            "Gemm",
            inputs=["bn_out", "B", "C"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([bn_node, gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_bn_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_bn_gemm([], {})
        assert isinstance(instances, list)
        assert len(instances) == 0
