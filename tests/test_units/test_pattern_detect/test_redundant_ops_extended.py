"""Extended tests for redundant operation detection."""

from typing import Any

import numpy as np
import pytest
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

    @pytest.mark.parametrize(
        ("inputs", "shape", "expected_count"),
        [
            (["Zero", "X"], (3,), 1),
            (["X", "Zero"], (2, 3), 1),
        ],
    )
    def test_zero_constant_input_detected(self, inputs, shape, expected_count):
        """Test detecting Add with zero as first or second input."""
        add_node = helper.make_node("Add", inputs=inputs, outputs=["Y"], name="add_0")
        nodes = [add_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(shape)),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == expected_count
        assert result[0]["shape"] == list(shape)

    def test_non_zero_constant_not_detected(self):
        """Test that non-zero Add is not detected."""
        add_node = helper.make_node("Add", inputs=["X", "One"], outputs=["Y"], name="add_0")
        nodes = [add_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 0

    def test_no_constant_input_not_detected(self):
        """Test Add without constant inputs."""
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        nodes = [add_node]
        initializers: dict[str, Any] = {}

        result = detect_add_zero(nodes, initializers)

        assert len(result) == 0

    def test_multiple_zero_adds_all_detected(self):
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

    def test_unnamed_node_uses_index_based_name(self):
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

    def test_second_input_zero_detected(self):
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

    def test_non_zero_not_detected(self):
        """Test that Sub with non-zero is not detected."""
        sub_node = helper.make_node("Sub", inputs=["X", "One"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 0

    def test_first_input_zero_not_detected(self):
        """Test that Sub with zero as first input is not detected (only checks second)."""
        sub_node = helper.make_node("Sub", inputs=["Zero", "X"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers = {
            "Zero": create_initializer("Zero", np.zeros(3)),
        }

        result = detect_sub_zero(nodes, initializers)

        # Should not detect because only checks second input
        assert len(result) == 0

    def test_insufficient_inputs_not_detected(self):
        """Test Sub with insufficient inputs."""
        sub_node = helper.make_node("Sub", inputs=["X"], outputs=["Y"], name="sub_0")
        nodes = [sub_node]
        initializers: dict[str, Any] = {}

        result = detect_sub_zero(nodes, initializers)

        assert len(result) == 0

    def test_multiple_zero_subs_all_detected(self):
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

    @pytest.mark.parametrize(
        ("inputs", "shape"),
        [
            (["One", "X"], (3,)),
            (["X", "One"], (2, 3)),
        ],
    )
    def test_one_constant_input_detected(self, inputs, shape):
        """Test detecting Mul with one as first or second input."""
        mul_node = helper.make_node("Mul", inputs=inputs, outputs=["Y"], name="mul_0")
        nodes = [mul_node]
        initializers = {
            "One": create_initializer("One", np.ones(shape)),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 1
        assert result[0]["shape"] == list(shape)

    def test_non_one_constant_not_detected(self):
        """Test that Mul with non-one is not detected."""
        mul_node = helper.make_node("Mul", inputs=["X", "Two"], outputs=["Y"], name="mul_0")
        nodes = [mul_node]
        initializers = {
            "Two": create_initializer("Two", 2 * np.ones(3)),
        }

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 0

    def test_no_constant_input_not_detected(self):
        """Test Mul without constant inputs."""
        mul_node = helper.make_node("Mul", inputs=["X", "Y"], outputs=["Z"], name="mul_0")
        nodes = [mul_node]
        initializers: dict[str, Any] = {}

        result = detect_mul_one(nodes, initializers)

        assert len(result) == 0

    def test_multiple_one_muls_all_detected(self):
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

    def test_second_input_one_detected(self):
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

    def test_non_one_constant_not_detected(self):
        """Test that Div with non-one is not detected."""
        div_node = helper.make_node("Div", inputs=["X", "Two"], outputs=["Y"], name="div_0")
        nodes = [div_node]
        initializers = {
            "Two": create_initializer("Two", 2 * np.ones(3)),
        }

        result = detect_div_one(nodes, initializers)

        assert len(result) == 0

    def test_first_input_one_not_detected(self):
        """Test that Div with one as first input is not detected (only checks second)."""
        div_node = helper.make_node("Div", inputs=["One", "X"], outputs=["Y"], name="div_0")
        nodes = [div_node]
        initializers = {
            "One": create_initializer("One", np.ones(3)),
        }

        result = detect_div_one(nodes, initializers)

        # Should not detect because only checks second input
        assert len(result) == 0

    def test_multiple_one_divs_all_detected(self):
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

    def test_all_zero_pads_detected(self):
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

    def test_non_zero_pads_not_detected(self):
        """Test that Pad with non-zero is not detected."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers = {
            "Pads": create_initializer("Pads", np.array([1, 1, 0, 0, 0, 0, 1, 1])),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 0

    @pytest.mark.parametrize(
        ("inputs", "init_dict"),
        [
            (["X"], {}),
            (["X", "Pads"], {}),
        ],
    )
    def test_missing_pads_initializer_not_detected(self, inputs, init_dict):
        """Test Pad without pads input or pads not in initializers."""
        pad_node = helper.make_node("Pad", inputs=inputs, outputs=["Y"], name="pad_0")
        nodes = [pad_node]
        initializers: dict[str, Any] = init_dict

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 0

    def test_unnamed_node_uses_index_based_name(self):
        """Test Pad without name uses index-based naming."""
        pad_node = helper.make_node("Pad", inputs=["X", "Pads"], outputs=["Y"])
        nodes = [pad_node]
        initializers = {
            "Pads": create_initializer("Pads", np.zeros(8)),
        }

        result = detect_pad_zero(nodes, initializers)

        assert len(result) == 1
        assert result[0]["node"] == "Pad_0"

    def test_multiple_zero_pads_all_detected(self):
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

    @pytest.mark.parametrize(
        ("op_type", "inputs", "shape"),
        [
            ("Reshape", ["X", "Shape"], [2, 3, 4]),
            ("Flatten", ["X"], [2, 3]),
        ],
    )
    def test_same_input_output_shape_detected(self, op_type, inputs, shape):
        """Test detecting Reshape/Flatten where input and output shapes match."""
        node = helper.make_node(op_type, inputs=inputs, outputs=["Y"], name="node_0")
        nodes = [node]
        data_shapes = {
            "X": shape,
            "Y": shape,
        }

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 1
        assert result[0]["op_type"] == op_type

    @pytest.mark.parametrize(
        ("data_shapes", "expected_count"),
        [
            pytest.param({"X": [2, 3, 4], "Y": [6, 4]}, 0, id="different_shapes"),
            pytest.param({"X": [], "Y": []}, 1, id="scalar_shapes"),
        ],
    )
    def test_reshape_detection_scenarios(self, data_shapes, expected_count):
        """Test identity Reshape detection with various shape configurations."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == expected_count
        # [REVIEW] Merged into test_reshape_detection_scenarios via parametrize — original: test_different_shapes_not_detected, test_scalar_shape_detected

    @pytest.mark.parametrize(
        "data_shapes",
        [
            {"Y": [2, 3, 4]},  # missing input shape
            {"X": [2, 3, 4]},  # missing output shape
        ],
    )
    def test_missing_shape_not_detected(self, data_shapes):
        """Test Reshape where input or output shape is missing."""
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0"
        )
        nodes = [reshape_node]

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_no_inputs_or_outputs_not_detected(self):
        """Test Reshape with no inputs or outputs."""
        reshape_node = helper.make_node("Reshape", inputs=[], outputs=[], name="reshape_0")
        nodes = [reshape_node]
        data_shapes: dict[str, int | list[int]] = {}

        result = detect_identity_reshape(nodes, data_shapes)

        assert len(result) == 0

    def test_unnamed_node_uses_index_based_name(self):
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

    def test_multiple_identity_ops_all_detected(self):
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
