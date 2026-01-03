"""Tests for constant to initializer conversion."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx._cst2initer import _constant_to_initializer

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConstantToInitializer:
    """Test _constant_to_initializer function."""

    def test_convert_constant_node_basic(self):
        """Test basic Constant node to initializer conversion."""
        # Create a Constant node with a simple float array
        constant_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        constant_tensor = numpy_helper.from_array(constant_value, "const_tensor")

        # Create a Constant node
        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output"],
            value=constant_tensor,
        )

        initializers: dict[str, Any] = {}
        nodes = [constant_node]

        # Convert constant node to initializer
        result_nodes = _constant_to_initializer(nodes, initializers)

        # Assert: Constant node should be removed from nodes
        assert len(result_nodes) == 0
        # Assert: Constant should be added to initializers
        assert "const_output" in initializers
        assert np.array_equal(numpy_helper.to_array(initializers["const_output"]), constant_value)

    def test_preserve_non_constant_nodes(self):
        """Test that non-Constant nodes are preserved."""
        # Create a regular node (e.g., Add)
        add_node = helper.make_node("Add", inputs=["input1", "input2"], outputs=["output"])

        initializers: dict[str, Any] = {}
        nodes = [add_node]

        # Convert (should not affect non-Constant nodes)
        result_nodes = _constant_to_initializer(nodes, initializers)

        # Assert: Non-constant nodes should be preserved
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Add"
        # Assert: No initializers added
        assert len(initializers) == 0

    def test_multiple_constant_nodes(self):
        """Test conversion of multiple Constant nodes."""
        # Create multiple constant nodes
        constant_value1 = np.array([1.0, 2.0], dtype=np.float32)
        constant_tensor1 = numpy_helper.from_array(constant_value1, "const_tensor1")
        constant_node1 = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output1"],
            value=constant_tensor1,
        )

        constant_value2 = np.array([3.0, 4.0], dtype=np.float32)
        constant_tensor2 = numpy_helper.from_array(constant_value2, "const_tensor2")
        constant_node2 = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output2"],
            value=constant_tensor2,
        )

        initializers: dict[str, Any] = {}
        nodes = [constant_node1, constant_node2]

        # Convert constants to initializers
        result_nodes = _constant_to_initializer(nodes, initializers)

        # Assert: Both constants should be removed
        assert len(result_nodes) == 0
        # Assert: Both should be in initializers
        assert "const_output1" in initializers
        assert "const_output2" in initializers
        assert len(initializers) == 2

    def test_constant_with_downstream_usage(self):
        """Test Constant converted and used by other nodes."""
        # Create constant node
        constant_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        constant_tensor = numpy_helper.from_array(constant_value, "const_tensor")
        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output"],
            value=constant_tensor,
        )

        # Create node that uses the constant
        add_node = helper.make_node("Add", inputs=["input", "const_output"], outputs=["result"])

        initializers: dict[str, Any] = {}
        nodes = [constant_node, add_node]

        # Convert
        result_nodes = _constant_to_initializer(nodes, initializers)

        # Assert: Constant removed, Add preserved
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Add"
        # Assert: Constant converted to initializer
        assert "const_output" in initializers

    def test_constant_int_type(self):
        """Test Constant node with integer data type."""
        constant_value = np.array([1, 2, 3], dtype=np.int64)
        constant_tensor = numpy_helper.from_array(constant_value, "const_tensor")
        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output"],
            value=constant_tensor,
        )

        initializers: dict[str, Any] = {}
        nodes = [constant_node]

        result_nodes = _constant_to_initializer(nodes, initializers)

        assert len(result_nodes) == 0
        assert "const_output" in initializers
        assert np.array_equal(numpy_helper.to_array(initializers["const_output"]), constant_value)

    def test_mixed_constant_and_non_constant_nodes(self):
        """Test model with mix of Constant and non-Constant nodes."""
        # Create constant node
        constant_value = np.array([1.0, 2.0], dtype=np.float32)
        constant_tensor = numpy_helper.from_array(constant_value, "const_tensor")
        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_output"],
            value=constant_tensor,
        )

        # Create regular nodes
        relu_node = helper.make_node("Relu", inputs=["input1"], outputs=["relu_output"])
        add_node = helper.make_node(
            "Add",
            inputs=["relu_output", "const_output"],
            outputs=["final_output"],
        )

        initializers: dict[str, Any] = {}
        nodes = [constant_node, relu_node, add_node]

        result_nodes = _constant_to_initializer(nodes, initializers)

        # Assert: Constant removed, others preserved
        assert len(result_nodes) == 2
        assert result_nodes[0].op_type == "Relu"
        assert result_nodes[1].op_type == "Add"
        # Assert: Constant converted to initializer
        assert "const_output" in initializers

    def test_empty_nodes_list(self):
        """Test with empty nodes list."""
        initializers: dict[str, Any] = {}
        nodes: list[Any] = []

        result_nodes = _constant_to_initializer(nodes, initializers)

        assert len(result_nodes) == 0
        assert len(initializers) == 0
