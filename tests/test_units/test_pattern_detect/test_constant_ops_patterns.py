"""Tests for constant foldable operation pattern detection."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from slimonnx.pattern_detect.constant_ops import detect_constant_foldable


class TestDetectConstantFoldable:
    """Test detect_constant_foldable function."""

    @pytest.mark.parametrize(
        ("nodes", "initializers", "expected_count"),
        [
            ([], {}, 0),  # empty
            (
                [
                    helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=["const_out"],
                        value=numpy_helper.from_array(
                            np.array([1.0, 2.0, 3.0], dtype=np.float32), "const_out"
                        ),
                    )
                ],
                {},
                0,
            ),  # constant only
        ],
    )
    def test_empty_and_constant_nodes(self, nodes, initializers, expected_count):
        """Test detection with empty or constant-only nodes."""
        result = detect_constant_foldable(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == expected_count

    def test_detects_shape_of_constant(self):
        """Test detection of Shape operation with constant input."""
        const_tensor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        const_init = numpy_helper.from_array(const_tensor, "input_const")

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_out"],
            value=const_init,
        )

        shape_node = helper.make_node(
            "Shape", inputs=["const_out"], outputs=["shape_out"], name="shape_0"
        )

        nodes = [const_node, shape_node]
        result = detect_constant_foldable(nodes, {})

        assert len(result) == 1
        assert result[0]["op_type"] == "Shape"
        assert result[0]["node"] == "shape_0"
        assert result[0]["can_fold"] is True

    def test_detects_unary_ops_with_constant_inputs(self):
        """Test detection with operation using initializer as input."""
        W = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        neg_node = helper.make_node("Neg", inputs=["W"], outputs=["neg_out"], name="neg_0")
        nodes = [neg_node]

        result = detect_constant_foldable(nodes, initializers)

        assert len(result) == 1
        assert result[0]["op_type"] == "Neg"
        assert result[0]["can_fold"] is True

    def test_skips_ops_with_non_constant_inputs(self):
        """Test that operations with non-constant inputs are not detected."""
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        nodes = [add_node]

        result = detect_constant_foldable(nodes, {})

        assert len(result) == 0

    def test_skips_unsupported_op_types(self):
        """Test that unsupported operation types are not detected."""
        W = np.ones(3, dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        conv_node = helper.make_node("Conv", inputs=["W", "W"], outputs=["conv_out"], name="conv_0")
        nodes = [conv_node]

        result = detect_constant_foldable(nodes, initializers)

        assert len(result) == 0

    @pytest.mark.parametrize(
        ("op_type", "inputs", "init_keys"),
        [
            ("Mul", ["W1", "W2"], ["W1", "W2"]),
            ("Sub", ["W1", "W2"], ["W1", "W2"]),
            ("Div", ["W1", "W2"], ["W1", "W2"]),
            ("Equal", ["W1", "W2"], ["W1", "W2"]),
        ],
    )
    def test_detects_binary_foldable_ops(self, op_type, inputs, init_keys):
        """Test detection of binary foldable operations with all constant inputs."""
        W1 = (
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
            if op_type != "Div"
            else np.array([10.0, 20.0, 30.0], dtype=np.float32)
        )
        W2 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
        }

        node = helper.make_node(
            op_type, inputs=inputs, outputs=["out"], name=f"{op_type.lower()}_0"
        )
        result = detect_constant_foldable([node], initializers)

        assert len(result) == 1
        assert result[0]["op_type"] == op_type

    def test_detects_cast_with_constant(self):
        """Test detection of Cast operation with constant input."""
        W = np.array([1, 2, 3], dtype=np.int32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        cast_node = helper.make_node(
            "Cast", inputs=["W"], outputs=["cast_out"], to=TensorProto.FLOAT, name="cast_0"
        )
        result = detect_constant_foldable([cast_node], initializers)

        assert len(result) == 1
        assert result[0]["op_type"] == "Cast"

    def test_detects_multiple_independent_foldable_ops(self):
        """Test detection of multiple foldable operations with independent constant inputs."""
        W1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        W2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        W3 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        initializers = {
            "W1": numpy_helper.from_array(W1, "W1"),
            "W2": numpy_helper.from_array(W2, "W2"),
            "W3": numpy_helper.from_array(W3, "W3"),
        }

        mul_node = helper.make_node("Mul", inputs=["W1", "W2"], outputs=["mul_out"], name="mul_0")
        neg_node = helper.make_node("Neg", inputs=["W3"], outputs=["neg_out"], name="neg_0")

        result = detect_constant_foldable([mul_node, neg_node], initializers)

        assert len(result) == 2
        assert result[0]["op_type"] == "Mul"
        assert result[1]["op_type"] == "Neg"

    def test_skips_ops_with_mixed_constant_nonconstant_inputs(self):
        """Test that operations with mixed constant/non-constant inputs are not foldable."""
        W = np.ones(3, dtype=np.float32)
        initializers = {"W": numpy_helper.from_array(W, "W")}

        add_node = helper.make_node("Add", inputs=["W", "X"], outputs=["add_out"], name="add_0")
        result = detect_constant_foldable([add_node], initializers)

        assert len(result) == 0

    def test_detects_matmul_with_constants(self):
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
        result = detect_constant_foldable([matmul_node], initializers)

        assert len(result) == 1
        assert result[0]["op_type"] == "MatMul"

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
