"""Tests for constant foldable operation pattern detection."""

from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from slimonnx.pattern_detect.constant_ops import detect_constant_foldable


class TestDetectConstantFoldable:
    """Test detect_constant_foldable function."""

    def test_detect_constant_foldable_empty_nodes(self):
        """Test detection with empty node list."""
        initializers: dict[str, Any] = {}
        nodes: list[Any] = []

        result = detect_constant_foldable(nodes, initializers)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_constant_node_only(self):
        """Test detection with only Constant node (should be skipped)."""
        constant_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        constant_tensor = numpy_helper.from_array(constant_value, "const_out")

        constant_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_out"],
            value=constant_tensor,
        )

        initializers: dict[str, Any] = {}
        nodes = [constant_node]

        result = detect_constant_foldable(nodes, initializers)

        # Constant nodes are skipped in results but tracked as available constants
        assert len(result) == 0

    def test_detect_constant_with_shape_operation(self):
        """Test detection of Shape operation with constant input."""
        # Create a Constant node whose output will be used by Shape
        const_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        const_init = numpy_helper.from_array(const_tensor, "input_const")

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_out"],
            value=const_init,
        )

        # Shape node using constant output
        shape_node = helper.make_node(
            "Shape", inputs=["const_out"], outputs=["shape_out"], name="shape_0"
        )

        initializers: dict[str, Any] = {}
        nodes = [const_node, shape_node]

        result = detect_constant_foldable(nodes, initializers)

        # Shape node should be detected as foldable (input is constant)
        assert len(result) == 1
        assert result[0]["op_type"] == "Shape"
        assert result[0]["node"] == "shape_0"
        assert result[0]["can_fold"] is True

    def test_detect_constant_with_initializer_input(self):
        """Test detection with operation using initializer as input."""
        W = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        # Neg operation on initializer (constant input)
        neg_node = helper.make_node("Neg", inputs=["W"], outputs=["neg_out"], name="neg_0")

        nodes = [neg_node]

        result = detect_constant_foldable(nodes, initializers)

        # Neg node should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Neg"
        assert result[0]["can_fold"] is True

    def test_detect_non_constant_input_not_foldable(self):
        """Test that operations with non-constant inputs are not detected."""
        initializers: dict[str, Any] = {}

        # Add node with non-constant inputs
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")

        nodes = [add_node]

        result = detect_constant_foldable(nodes, initializers)

        # Add node should not be detected (inputs are not constants)
        assert len(result) == 0

    def test_detect_unsupported_op_not_foldable(self):
        """Test that unsupported operation types are not detected."""
        W = np.ones(3, dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        # Conv operation (not in FOLDABLE_OP_TYPES)
        conv_node = helper.make_node("Conv", inputs=["W", "W"], outputs=["conv_out"], name="conv_0")

        nodes = [conv_node]

        result = detect_constant_foldable(nodes, initializers)

        # Conv should not be detected (not a foldable operation)
        assert len(result) == 0

    def test_detect_mul_with_all_constant_inputs(self):
        """Test detection of Mul with all constant inputs."""
        W1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        mul_node = helper.make_node("Mul", inputs=["W1", "W2"], outputs=["mul_out"], name="mul_0")

        nodes = [mul_node]

        result = detect_constant_foldable(nodes, initializers)

        # Mul should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Mul"

    def test_detect_cast_with_constant_input(self):
        """Test detection of Cast operation with constant input."""
        W = np.array([1, 2, 3], dtype=np.int32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        cast_node = helper.make_node(
            "Cast",
            inputs=["W"],
            outputs=["cast_out"],
            to=TensorProto.FLOAT,
            name="cast_0",
        )

        nodes = [cast_node]

        result = detect_constant_foldable(nodes, initializers)

        # Cast should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Cast"

    def test_detect_multiple_foldable_operations(self):
        """Test detection of multiple foldable operations with independent constant inputs."""
        W1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        W3 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
            "W3": numpy_helper.from_array(W3, "W3"),
        }

        # First: Mul(W1, W2) -> mul_out
        mul_node = helper.make_node("Mul", inputs=["W1", "W2"], outputs=["mul_out"], name="mul_0")

        # Second: Neg(W3) -> neg_out (Independent operation with constant input)
        neg_node = helper.make_node("Neg", inputs=["W3"], outputs=["neg_out"], name="neg_0")

        nodes = [mul_node, neg_node]

        result = detect_constant_foldable(nodes, initializers)

        # Both should be detected as foldable (they have independent constant inputs)
        assert len(result) == 2
        assert result[0]["op_type"] == "Mul"
        assert result[1]["op_type"] == "Neg"

    def test_detect_partial_constant_inputs_not_foldable(self):
        """Test that operations with mixed constant/non-constant inputs are not foldable."""
        W = np.ones(3, dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        # Add with one constant input and one non-constant input
        add_node = helper.make_node("Add", inputs=["W", "X"], outputs=["add_out"], name="add_0")

        nodes = [add_node]

        result = detect_constant_foldable(nodes, initializers)

        # Should not be foldable (X is not constant)
        assert len(result) == 0

    def test_detect_sub_with_constant_inputs(self):
        """Test detection of Sub operation with constant inputs."""
        W1 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        W2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        sub_node = helper.make_node("Sub", inputs=["W1", "W2"], outputs=["sub_out"], name="sub_0")

        nodes = [sub_node]

        result = detect_constant_foldable(nodes, initializers)

        # Sub should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Sub"

    def test_detect_div_with_constant_inputs(self):
        """Test detection of Div operation with constant inputs."""
        W1 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        W2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        div_node = helper.make_node("Div", inputs=["W1", "W2"], outputs=["div_out"], name="div_0")

        nodes = [div_node]

        result = detect_constant_foldable(nodes, initializers)

        # Div should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Div"

    def test_detect_matmul_with_constant_inputs(self):
        """Test detection of MatMul operation with constant inputs."""
        W1 = np.random.randn(3, 4).astype(np.float32)
        W2 = np.random.randn(4, 2).astype(np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        matmul_node = helper.make_node(
            "MatMul", inputs=["W1", "W2"], outputs=["matmul_out"], name="matmul_0"
        )

        nodes = [matmul_node]

        result = detect_constant_foldable(nodes, initializers)

        # MatMul should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "MatMul"

    def test_detect_equal_with_constant_inputs(self):
        """Test detection of Equal operation with constant inputs."""
        W1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        equal_node = helper.make_node(
            "Equal", inputs=["W1", "W2"], outputs=["equal_out"], name="equal_0"
        )

        nodes = [equal_node]

        result = detect_constant_foldable(nodes, initializers)

        # Equal should be detected as foldable
        assert len(result) == 1
        assert result[0]["op_type"] == "Equal"

    def test_detect_result_structure(self):
        """Test that result has correct structure."""
        W = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        neg_node = helper.make_node("Neg", inputs=["W"], outputs=["neg_out"], name="neg_0")

        nodes = [neg_node]

        result = detect_constant_foldable(nodes, initializers)

        # Check result structure
        assert len(result) == 1
        assert "node" in result[0]
        assert "op_type" in result[0]
        assert "inputs" in result[0]
        assert "outputs" in result[0]
        assert "can_fold" in result[0]

        assert result[0]["node"] == "neg_0"
        assert result[0]["op_type"] == "Neg"
        assert result[0]["inputs"] == ["W"]
        assert result[0]["outputs"] == ["neg_out"]
        assert result[0]["can_fold"] is True
