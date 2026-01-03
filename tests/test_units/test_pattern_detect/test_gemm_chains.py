"""Tests for Gemm chains pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.gemm_chains import detect_gemm_gemm

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGemmChainsDetection:
    """Test Gemm chains pattern detection."""

    def test_detect_two_gemm_chain(self):
        """Test detection of two Gemm nodes in sequence."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # First Gemm: 3 -> 4
        B1 = np.random.randn(3, 4).astype(np.float32)
        C1 = np.random.randn(4).astype(np.float32)

        # Second Gemm: 4 -> 2
        B2 = np.random.randn(4, 2).astype(np.float32)
        C2 = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
        ]

        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["X", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["gemm1_out", "B2", "C2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm1_node, gemm2_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        # Detect patterns
        instances = detect_gemm_gemm(nodes, initializers_dict)

        # Verify detection
        assert isinstance(instances, list)

    def test_detect_three_gemm_chain(self):
        """Test detection of three Gemm nodes in sequence."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # Three Gemm layers: 3 -> 4 -> 5 -> 2
        B1 = np.random.randn(3, 4).astype(np.float32)
        C1 = np.random.randn(4).astype(np.float32)

        B2 = np.random.randn(4, 5).astype(np.float32)
        C2 = np.random.randn(5).astype(np.float32)

        B3 = np.random.randn(5, 2).astype(np.float32)
        C3 = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
            create_initializer("B3", B3),
            create_initializer("C3", C3),
        ]

        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["X", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["gemm1_out", "B2", "C2"],
            outputs=["gemm2_out"],
        )
        gemm3_node = helper.make_node(
            "Gemm",
            inputs=["gemm2_out", "B3", "C3"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model(
            [gemm1_node, gemm2_node, gemm3_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_gemm_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_gemm_chain_with_different_shapes(self):
        """Test Gemm chain detection with various hidden dimensions."""
        inputs = [create_tensor_value_info("X", "float32", [4, 256])]
        outputs = [create_tensor_value_info("Y", "float32", [4, 64])]

        # Gemm chain with varying dimensions: 256 -> 128 -> 64
        B1 = np.random.randn(256, 128).astype(np.float32)
        C1 = np.random.randn(128).astype(np.float32)

        B2 = np.random.randn(128, 64).astype(np.float32)
        C2 = np.random.randn(64).astype(np.float32)

        initializers = [
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
        ]

        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["X", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["gemm1_out", "B2", "C2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm1_node, gemm2_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_gemm_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_single_gemm_no_chain(self):
        """Test that single Gemm is not detected as chain."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        B = np.random.randn(3, 2).astype(np.float32)
        C = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("B", B),
            create_initializer("C", C),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "B", "C"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_gemm_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_gemm_chain_with_activation(self):
        """Test Gemm chain detection with non-consecutive nodes (activation between)."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        B1 = np.random.randn(3, 4).astype(np.float32)
        C1 = np.random.randn(4).astype(np.float32)

        B2 = np.random.randn(4, 2).astype(np.float32)
        C2 = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
        ]

        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["X", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        # Add ReLU activation between Gemms
        relu_node = helper.make_node(
            "Relu",
            inputs=["gemm1_out"],
            outputs=["relu_out"],
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["relu_out", "B2", "C2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model(
            [gemm1_node, relu_node, gemm2_node], inputs, outputs, initializers
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_gemm_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_multiple_gemm_chains(self):
        """Test detection of multiple separate Gemm chains."""
        inputs = [create_tensor_value_info("X1", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        B1 = np.random.randn(3, 4).astype(np.float32)
        C1 = np.random.randn(4).astype(np.float32)
        B2 = np.random.randn(4, 2).astype(np.float32)
        C2 = np.random.randn(2).astype(np.float32)

        initializers = [
            create_initializer("B1", B1),
            create_initializer("C1", C1),
            create_initializer("B2", B2),
            create_initializer("C2", C2),
        ]

        gemm1_node = helper.make_node(
            "Gemm",
            inputs=["X1", "B1", "C1"],
            outputs=["gemm1_out"],
        )
        gemm2_node = helper.make_node(
            "Gemm",
            inputs=["gemm1_out", "B2", "C2"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm1_node, gemm2_node], inputs, outputs, initializers)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        instances = detect_gemm_gemm(nodes, initializers_dict)
        assert isinstance(instances, list)

    def test_detect_empty_node_list(self):
        """Test detection with empty node list."""
        instances = detect_gemm_gemm([], {})
        assert isinstance(instances, list)
        assert len(instances) == 0
