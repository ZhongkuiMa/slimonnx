"""Extended tests for redundant operation detection."""

from typing import Any

import numpy as np
from onnx import helper, numpy_helper

from slimonnx.pattern_detect.redundant_ops import (
    detect_add_zero,
    detect_div_one,
    detect_identity_reshape,
    detect_mul_one,
    detect_pad_zero,
    detect_sub_zero,
)


def create_initializer(name, array):
    """Create a TensorProto initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


class TestDetectAddZero:
    """Test detect_add_zero function."""

    def test_detect_add_zero_basic(self):
        """Test detecting Add with zero as first input."""
        add_node = helper.make_node("Add", inputs=["Zero", "X"], outputs=["Y"], name="add_0")
        nodes = [add_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(3)),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "add_0"
        assert result[0]["initializer"] == "Zero"
        assert result[0]["shape"] == [3]

    def test_detect_add_zero_second_input(self):
        """Test detecting Add with zero as second input."""
        add_node = helper.make_node("Add", inputs=["X", "Zero"], outputs=["Y"], name="add_1")
        nodes = [add_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros((2, 3))),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["shape"] == [2, 3]

    def test_detect_add_non_zero(self):
        """Test that non-zero Add is not detected."""
        add_node = helper.make_node("Add", inputs=["X", "One"], outputs=["Y"], name="add_0")
        nodes = [add_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_add_no_constant(self):
        """Test Add without constant inputs."""
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        nodes = [add_node]
        initializers: dict[str, Any] = {}

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_add_multiple_zeros(self):
        """Test detecting multiple Add operations with zero."""
        add_node1 = helper.make_node("Add", inputs=["Zero1", "X"], outputs=["Y"], name="add_0")
        add_node2 = helper.make_node("Add", inputs=["A", "Zero2"], outputs=["B"], name="add_1")
        add_node3 = helper.make_node("Add", inputs=["C", "D"], outputs=["E"], name="add_2")

        nodes = [add_node1, add_node2, add_node3]
        initializers = {
            "Zero1": create_initializer("Zero1", np.zeros(2)),
            "Zero2": create_initializer("Zero2", np.zeros((3, 3))),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 2

    def test_detect_add_unnamed_node(self):
        """Test Add without name uses index-based naming."""
        add_node = helper.make_node("Add", inputs=["Zero", "X"], outputs=["Y"])
        nodes = [add_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(3)),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "Add_0"


class TestDetectSubZero:
    """Test detect_sub_zero function."""

    def test_detect_sub_zero_basic(self):
        """Test detecting Sub with zero as second input."""
        sub_node = helper.make_node("Sub", inputs=["X", "Zero"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(3)),
        }

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "sub_0"
        assert result[0]["initializer"] == "Zero"

    def test_detect_sub_non_zero(self):
        """Test that Sub with non-zero is not detected."""
        sub_node = helper.make_node("Sub", inputs=["X", "One"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_sub_zero_first_input(self):
        """Test that Sub with zero as first input is not detected (only checks second)."""
        sub_node = helper.make_node("Sub", inputs=["Zero", "X"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(3)),
        }

        result = detect_sub_zero(nodes, initializers)

        # Should not detect because only checks second input
        assert len(result) == 0

    def test_detect_sub_no_inputs(self):
        """Test Sub with insufficient inputs."""
        sub_node = helper.make_node("Sub", inputs=["X"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers: dict[str, Any] = {}

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_sub_multiple(self):
        """Test detecting multiple Sub operations."""
        sub_node1 = helper.make_node("Sub", inputs=["X", "Zero1"], outputs=["Y"], name="sub_0")
        sub_node2 = helper.make_node("Sub", inputs=["A", "Zero2"], outputs=["B"], name="sub_1")

        nodes = [sub_node1, sub_node2]
        initializers = {
            "Zero1": create_initializer("Zero1", np.zeros(2)),
            "Zero2": create_initializer("Zero2", np.zeros((3, 3))),
        }

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 2


class TestDetectMulOne:
    """Test detect_mul_one function."""

    def test_detect_mul_one_basic(self):
        """Test detecting Mul with one as first input."""
        mul_node = helper.make_node("Mul", inputs=["One", "X"], outputs=["Y"], name="mul_0")
        nodes = [mul_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "mul_0"
        assert result[0]["initializer"] == "One"

    def test_detect_mul_one_second_input(self):
        """Test detecting Mul with one as second input."""
        mul_node = helper.make_node("Mul", inputs=["X", "One"], outputs=["Y"], name="mul_0")
        nodes = [mul_node]
        initializers = {
            "One": create_initializer("One", np.ones((2, 3))),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 1
        assert result[0]["shape"] == [2, 3]

    def test_detect_mul_non_one(self):
        """Test that Mul with non-one is not detected."""
        mul_node = helper.make_node("Mul", inputs=["X", "Two"], outputs=["Y"], name="mul_0")
        nodes = [mul_node]
        initializers = {
            "Two": create_initializer("Two", 2 * np.ones(3)),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 0

    def test_detect_mul_no_constant(self):
        """Test Mul without constant inputs."""
        mul_node = helper.make_node("Mul", inputs=["X", "Y"], outputs=["Z"], name="mul_0")
        nodes = [mul_node]
        initializers: dict[str, Any] = {}

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 0

    def test_detect_mul_multiple(self):
        """Test detecting multiple Mul operations."""
        mul_node1 = helper.make_node("Mul", inputs=["One1", "X"], outputs=["Y"], name="mul_0")
        mul_node2 = helper.make_node("Mul", inputs=["A", "One2"], outputs=["B"], name="mul_1")

        nodes = [mul_node1, mul_node2]
        initializers = {
            "One1": create_initializer("One1", np.ones(2)),
            "One2": create_initializer("One2", np.ones((3, 3))),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 2


class TestDetectDivOne:
    """Test detect_div_one function."""

    def test_detect_div_one_basic(self):
        """Test detecting Div with one as second input."""
        div_node = helper.make_node("Div", inputs=["X", "One"], outputs=["Y"], name="div_0")
        nodes = [div_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_div_one(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "div_0"
        assert result[0]["initializer"] == "One"

    def test_detect_div_non_one(self):
        """Test that Div with non-one is not detected."""
        div_node = helper.make_node("Div", inputs=["X", "Two"], outputs=["Y"], name="div_0")
        nodes = [div_node]
        initializers = {
            "Two": create_initializer("Two", 2 * np.ones(3)),
        }

        result = detect_div_one(nodes, initializers)

        assert len(result) == 0

    def test_detect_div_one_first_input(self):
        """Test that Div with one as first input is not detected (only checks second)."""
        div_node = helper.make_node("Div", inputs=["One", "X"], outputs=["Y"], name="div_0")
        nodes = [div_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_div_one(nodes, initializers)

        # Should not detect because only checks second input
        assert len(result) == 0

    def test_detect_div_multiple(self):
        """Test detecting multiple Div operations."""
        div_node1 = helper.make_node("Div", inputs=["X", "One1"], outputs=["Y"], name="div_0")
        div_node2 = helper.make_node("Div", inputs=["A", "One2"], outputs=["B"], name="div_1")

        nodes = [div_node1, div_node2]
        initializers = {
            "One1": create_initializer("One1", np.ones(2)),
            "One2": create_initializer("One2", np.ones((3, 3))),
        }

        result = detect_div_one(nodes, initializers)

        assert len(result) == 2


class TestDetectPadZero:
    """Test detect_pad_zero function."""

    def test_detect_pad_zero_basic(self):
        """Test detecting Pad with zero padding."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers = {
            "Pads": create_initializer("Pads", np.zeros(8)),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "pad_0"
        assert result[0]["pads"] == [0.0] * 8

    def test_detect_pad_non_zero(self):
        """Test that Pad with non-zero is not detected."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers = {
            "Pads": create_initializer("Pads", np.array([1, 1, 0, 0, 0, 0, 1, 1])),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_pad_no_pads_input(self):
        """Test Pad without pads input."""
        pad_node = helper.make_node("Pad", inputs=["X"], outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers: dict[str, Any] = {}

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_pad_pads_not_in_initializers(self):
        """Test Pad where pads is not in initializers."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers: dict[str, Any] = {}

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 0

    def test_detect_pad_unnamed_node(self):
        """Test Pad without name uses index-based naming."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"])
        nodes = [pad_node]
        initializers = {
            "Pads": create_initializer("Pads", np.zeros(8)),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "Pad_0"

    def test_detect_pad_multiple(self):
        """Test detecting multiple Pad operations."""
        pad_node1 = helper.make_node("Pad", inputs=["X", "Pads1"], outputs=["Y"], name="pad_0")
        pad_node2 = helper.make_node("Pad", inputs=["A", "Pads2"], outputs=["B"], name="pad_1")

        nodes = [pad_node1, pad_node2]
        initializers = {
            "Pads1": create_initializer("Pads1", np.zeros(8)),
            "Pads2": create_initializer("Pads2", np.zeros(4)),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 2


class TestDetectIdentityReshape:
    """Test detect_identity_reshape function."""

    def test_detect_identity_reshape_same_shape(self):
        """Test detecting Reshape where input and output shapes match."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]
        data_shapes = {
            "X": [2, 3, 4],
            "Y": [2, 3, 4],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 1
        assert result[0]["node"] == "reshape_0"
        assert result[0]["op_type"] == "Reshape"
        assert result[0]["shape"] == [2, 3, 4]

    def test_detect_identity_reshape_different_shapes(self):
        """Test that Reshape with different shapes is not detected."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]
        data_shapes = {
            "X": [2, 3, 4],
            "Y": [6, 4],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_detect_identity_flatten_same_shape(self):
        """Test detecting Flatten where input and output shapes match."""
        flatten_node = helper.make_node("Flatten", inputs=["X"], outputs=["Y"], name="flatten_0")
        nodes = [flatten_node]
        data_shapes = {
            "X": [2, 3],
            "Y": [2, 3],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 1
        assert result[0]["op_type"] == "Flatten"

    def test_detect_identity_reshape_missing_input_shape(self):
        """Test Reshape where input shape is missing."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]
        data_shapes = {
            "Y": [2, 3, 4],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_detect_identity_reshape_missing_output_shape(self):
        """Test Reshape where output shape is missing."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]
        data_shapes = {
            "X": [2, 3, 4],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_detect_identity_reshape_no_inputs_or_outputs(self):
        """Test Reshape with no inputs or outputs."""
        reshape_node = helper.make_node("Reshape", inputs=[], outputs=[], name="reshape_0")
        nodes = [reshape_node]
        data_shapes: dict[str, int | list[int]] = {}

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_detect_identity_reshape_unnamed_node(self):
        """Test Reshape without name uses index-based naming."""
        reshape_node = helper.make_node("Reshape", inputs=["X", "Shape"], outputs=["Y"])
        nodes = [reshape_node]
        data_shapes = {
            "X": [2, 3],
            "Y": [2, 3],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 1
        assert result[0]["node"] == "Reshape_0"

    def test_detect_identity_reshape_multiple(self):
        """Test detecting multiple identity Reshape operations."""
        reshape_node1 = helper.make_node(
            "Reshape", inputs=["X", "Shape1"], outputs=["Y"], name="reshape_0"
        )
        reshape_node2 = helper.make_node(
            "Reshape", inputs=["A", "Shape2"], outputs=["B"], name="reshape_1"
        )
        flatten_node = helper.make_node("Flatten", inputs=["C"], outputs=["D"], name="flatten_0")

        nodes = [reshape_node1, reshape_node2, flatten_node]
        data_shapes = {
            "X": [2, 3],
            "Y": [2, 3],
            "A": [4, 5, 6],
            "B": [4, 5, 6],
            "C": [7, 8],
            "D": [7, 8],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 3

    def test_detect_identity_reshape_scalars(self):
        """Test identity Reshape with scalar shapes."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]
        data_shapes: dict[str, int | list[int]] = {
            "X": [],
            "Y": [],
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 1
