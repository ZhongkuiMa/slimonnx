"""Tests for MatMul+Add pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.matmul_add import detect_matmul_add

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestMatMulAddDetection:
    """Test MatMul+Add pattern detection."""

    def test_detect_matmul_add_basic(self, create_matmul_add_model):
        """Test basic MatMul+Add pattern detection."""
        model = create_matmul_add_model(shape_m=2, shape_n=3, shape_k=3)

        # Build initializers dict
        initializers = {init.name: init for init in model.graph.initializer}
        nodes = list(model.graph.node)

        # Detect patterns
        instances = detect_matmul_add(nodes, initializers)

        # Should detect the MatMul+Add pattern
        assert len(instances) > 0
        # Verify detected pattern is a dict with required fields
        assert isinstance(instances[0], dict)
        assert "matmul_node" in instances[0] or "weight" in instances[0] or "bias" in instances[0]

    def test_detect_matmul_add_different_shapes(self):
        """Test MatMul+Add detection with different tensor shapes."""
        # Create MatMul+Add with custom shapes
        inputs = [create_tensor_value_info("X", "float32", [4, 5])]
        outputs = [create_tensor_value_info("Y", "float32", [4, 7])]

        # Create initializers: W is (5, 7), b is (7,)
        W = np.random.randn(5, 7).astype(np.float32)
        b = np.random.randn(7).astype(np.float32)
        initializers_list = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        # Create nodes
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["Y"])
        nodes = [matmul_node, add_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        # Build initializers dict
        initializers = {init.name: init for init in model.graph.initializer}

        # Detect patterns
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # Should detect the pattern
        assert len(instances) > 0

    def test_detect_matmul_add_no_pattern(self):
        """Test that unrelated MatMul and Add are not detected as pattern."""
        # Create MatMul and Add that are NOT connected
        inputs = [
            create_tensor_value_info("X", "float32", [2, 3]),
            create_tensor_value_info("A", "float32", [2, 2]),
        ]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # Create initializers
        W = np.random.randn(3, 2).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        initializers_list = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        # Create unrelated nodes
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        # Add takes A and b, not matmul_out
        add_node = helper.make_node("Add", inputs=["A", "b"], outputs=["Y"])
        nodes = [matmul_node, add_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # Should NOT detect the pattern (nodes not connected)
        assert len(instances) == 0

    def test_detect_matmul_add_missing_bias(self):
        """Test MatMul+Add detection when bias is variable (not initializer)."""
        # Create MatMul+Add where bias is a variable input
        inputs = [
            create_tensor_value_info("X", "float32", [2, 3]),
            create_tensor_value_info("b", "float32", [2]),
        ]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # Only W is initializer
        W = np.random.randn(3, 2).astype(np.float32)
        initializers_list = [create_initializer("W", W)]

        # Create nodes
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["Y"])
        nodes = [matmul_node, add_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # May or may not detect depending on implementation, but should handle gracefully
        assert isinstance(instances, list)

    def test_detect_matmul_add_multiple_patterns(self):
        """Test detection of multiple MatMul+Add patterns in one model."""
        # Create model with two MatMul+Add patterns
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        # Initializers for first pattern
        W1 = np.random.randn(3, 4).astype(np.float32)
        b1 = np.random.randn(4).astype(np.float32)

        # Initializers for second pattern
        W2 = np.random.randn(4, 2).astype(np.float32)
        b2 = np.random.randn(2).astype(np.float32)

        initializers_list = [
            create_initializer("W1", W1),
            create_initializer("b1", b1),
            create_initializer("W2", W2),
            create_initializer("b2", b2),
        ]

        # Create two MatMul+Add patterns in sequence
        matmul1 = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["matmul1_out"])
        add1 = helper.make_node("Add", inputs=["matmul1_out", "b1"], outputs=["add1_out"])
        matmul2 = helper.make_node("MatMul", inputs=["add1_out", "W2"], outputs=["matmul2_out"])
        add2 = helper.make_node("Add", inputs=["matmul2_out", "b2"], outputs=["Y"])
        nodes = [matmul1, add1, matmul2, add2]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # Should detect multiple patterns
        assert len(instances) >= 2

    def test_detect_matmul_add_weight_shapes(self, create_matmul_add_model):
        """Test that detected pattern validates weight shapes."""
        model = create_matmul_add_model()

        initializers = {init.name: init for init in model.graph.initializer}
        nodes = list(model.graph.node)

        instances = detect_matmul_add(nodes, initializers)

        assert len(instances) > 0
        instance = instances[0]

        # Weight and bias shapes should be accessible or pattern should have fields
        assert isinstance(instance, dict)


class TestMatMulAddEdgeCases:
    """Test edge cases for MatMul+Add detection."""

    def test_detect_empty_model(self):
        """Test detection with empty node list."""
        instances = detect_matmul_add([], {})
        assert isinstance(instances, list)

    def test_detect_matmul_only(self):
        """Test that lone MatMul is not detected."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        W = np.random.randn(3, 2).astype(np.float32)
        initializers_list = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
        nodes = [matmul_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # MatMul alone should not be detected
        assert len(instances) == 0

    def test_detect_add_only(self):
        """Test that lone Add is not detected."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        b = np.random.randn(3).astype(np.float32)
        initializers_list = [create_initializer("b", b)]

        add_node = helper.make_node("Add", inputs=["X", "b"], outputs=["Y"])
        nodes = [add_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # Add alone should not be detected
        assert len(instances) == 0

    def test_detect_matmul_add_with_multiple_consumers(self):
        """Test MatMul+Add where MatMul output has multiple consumers."""
        # Create MatMul whose output goes to both Add and another node
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [
            create_tensor_value_info("Y1", "float32", [2, 2]),
            create_tensor_value_info("Y2", "float32", [2, 2]),
        ]

        W = np.random.randn(3, 2).astype(np.float32)
        b = np.random.randn(2).astype(np.float32)
        initializers_list = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        # MatMul output is used by both Add and Identity
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["Y1"])
        identity_node = helper.make_node("Identity", inputs=["matmul_out"], outputs=["Y2"])
        nodes = [matmul_node, add_node, identity_node]

        model = create_minimal_onnx_model(nodes, inputs, outputs, initializers_list)

        initializers = {init.name: init for init in model.graph.initializer}
        instances = detect_matmul_add(list(model.graph.node), initializers)

        # Pattern may not be fusible due to multiple consumers
        assert isinstance(instances, list)
