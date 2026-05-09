"""Extended tests for constant folding operations (_cst_op.py)."""

from typing import Any

import numpy as np
import pytest
from onnx import helper, numpy_helper

from slimonnx.optimize_onnx._cst_op import (
    _can_fold_node,
    _execute_aggregation_ops,
    _execute_binary_arithmetic,
    _execute_concat,
    _execute_elementwise_ops,
    _execute_generation_ops,
    _execute_shape_manipulation_ops,
    _execute_type_and_logic_ops,
    _fuse_constant_nodes,
)


def create_initializer(name, array):
    """Create a TensorProto initializer from numpy array."""
    return numpy_helper.from_array(array.astype(np.float32), name)


def create_initializer_int64(name, array):
    """Create a TensorProto initializer with int64 dtype."""
    return numpy_helper.from_array(array.astype(np.int64), name)


class TestCanFoldNode:
    """Test _can_fold_node function."""

    def test_succeeds_when_all_inputs_available(self):
        """Test folding when all inputs are in initializers."""
        node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        initializers = {
            "X": create_initializer("X", np.ones(3)),
            "Y": create_initializer("Y", np.ones(3)),
        }
        nodes_to_delete: list[Any] = []

        result = _can_fold_node(node, initializers, nodes_to_delete)

        assert result is True

    @pytest.mark.parametrize(
        ("nodes_to_delete", "expected"),
        [
            pytest.param([], False, id="missing_input_not_foldable"),
            pytest.param(["Y"], True, id="missing_in_deleted_nodes_foldable"),
        ],
    )
    def test_fold_with_missing_input(self, nodes_to_delete, expected):
        """Test fold decision when an input is missing from initializers."""
        node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        initializers = {
            "X": create_initializer("X", np.ones(3)),
        }

        result = _can_fold_node(node, initializers, nodes_to_delete)

        assert result is expected


class TestExecuteBinaryArithmetic:
    """Test _execute_binary_arithmetic function."""

    @pytest.mark.parametrize(
        ("op_type", "node_name", "x_vals", "y_vals", "expected"),
        [
            pytest.param(
                "Add", "add_0", [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0], id="add"
            ),
            pytest.param(
                "Sub", "sub_0", [5.0, 6.0, 7.0], [1.0, 2.0, 3.0], [4.0, 4.0, 4.0], id="sub"
            ),
            pytest.param(
                "Mul", "mul_0", [2.0, 3.0, 4.0], [2.0, 2.0, 2.0], [4.0, 6.0, 8.0], id="mul"
            ),
            pytest.param(
                "Div", "div_0", [4.0, 6.0, 8.0], [2.0, 2.0, 2.0], [2.0, 3.0, 4.0], id="div_float"
            ),
            pytest.param(
                "Pow", "pow_0", [2.0, 3.0, 4.0], [2.0, 2.0, 2.0], [4.0, 9.0, 16.0], id="pow"
            ),
        ],
    )
    def test_elementwise_binary_ops(self, op_type, node_name, x_vals, y_vals, expected):
        """Verify _execute_binary_arithmetic across elementwise binary operators."""
        node = helper.make_node(op_type, inputs=["X", "Y"], outputs=["Z"], name=node_name)
        initializers = {
            "X": create_initializer("X", np.array(x_vals)),
            "Y": create_initializer("Y", np.array(y_vals)),
        }
        result = _execute_binary_arithmetic(node, initializers)
        assert np.allclose(result, np.array(expected))

    def test_div_integer(self):
        """Test Div operation with integers (should use floor division)."""
        node = helper.make_node("Div", inputs=["X", "Y"], outputs=["Z"], name="div_0")
        x_array = np.array([5, 7, 9], dtype=np.int32)
        y_array = np.array([2, 2, 2], dtype=np.int32)
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Y": numpy_helper.from_array(y_array, "Y"),
        }

        result = _execute_binary_arithmetic(node, initializers)

        assert np.array_equal(result, np.array([2, 3, 4]))

    def test_matrix_multiplies_correctly(self):
        """Test MatMul produces correct product."""
        node = helper.make_node("MatMul", inputs=["X", "Y"], outputs=["Z"], name="mm_0")
        x_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_array = np.array([[5.0, 6.0], [7.0, 8.0]])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Y": numpy_helper.from_array(y_array, "Y"),
        }

        result = _execute_binary_arithmetic(node, initializers)

        expected = np.matmul(x_array, y_array)
        assert np.allclose(result, expected)


class TestExecuteElementwiseOps:
    """Test _execute_elementwise_ops function."""

    def test_clips_negative_to_zero(self):
        """Test Relu clips negative values to zero."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        initializers = {
            "X": create_initializer("X", np.array([-1.0, 0.0, 1.0, 2.0])),
        }

        result = _execute_elementwise_ops(node, initializers)

        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 2.0]))

    def test_negates_elementwise(self):
        """Test Neg negates all elements."""
        node = helper.make_node("Neg", inputs=["X"], outputs=["Y"], name="neg_0")
        initializers = {
            "X": create_initializer("X", np.array([1.0, -2.0, 3.0])),
        }

        result = _execute_elementwise_ops(node, initializers)

        assert np.allclose(result, np.array([-1.0, 2.0, -3.0]))

    def test_binary_arithmetic_via_elementwise(self):
        """Test binary arithmetic through elementwise ops."""
        node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        initializers = {
            "X": create_initializer("X", np.array([1.0, 2.0])),
            "Y": create_initializer("Y", np.array([3.0, 4.0])),
        }

        result = _execute_elementwise_ops(node, initializers)

        assert np.allclose(result, np.array([4.0, 6.0]))

    def test_unsupported_op(self):
        """Test unsupported operation returns None."""
        node = helper.make_node("Unsupported", inputs=["X"], outputs=["Y"], name="unsup_0")
        initializers: dict[str, Any] = {}

        result = _execute_elementwise_ops(node, initializers)

        assert result is None


class TestExecuteGenerationOps:
    """Test _execute_generation_ops function."""

    def test_range_basic(self):
        """Test Range operation."""
        node = helper.make_node(
            "Range", inputs=["Start", "Limit", "Delta"], outputs=["Y"], name="range_0"
        )
        initializers = {
            "Start": create_initializer("Start", np.array([0.0])),
            "Limit": create_initializer("Limit", np.array([10.0])),
            "Delta": create_initializer("Delta", np.array([2.0])),
        }

        result = _execute_generation_ops(node, initializers)

        assert np.allclose(result, np.arange(0, 10, 2))

    def test_range_missing_input(self):
        """Test Range returns None if input is missing."""
        node = helper.make_node(
            "Range", inputs=["Start", "Limit", "Delta"], outputs=["Y"], name="range_0"
        )
        initializers = {
            "Start": create_initializer("Start", np.array([0.0])),
        }

        result = _execute_generation_ops(node, initializers)

        assert result is None

    def test_range_non_scalar_input(self):
        """Test Range returns None if input is not scalar."""
        node = helper.make_node(
            "Range", inputs=["Start", "Limit", "Delta"], outputs=["Y"], name="range_0"
        )
        initializers = {
            "Start": create_initializer("Start", np.array([0.0, 1.0])),
            "Limit": create_initializer("Limit", np.array([10.0])),
            "Delta": create_initializer("Delta", np.array([2.0])),
        }

        result = _execute_generation_ops(node, initializers)

        assert result is None

    def test_unsupported_generation_op(self):
        """Test unsupported generation op returns None."""
        node = helper.make_node("UnsupportedGen", inputs=["X"], outputs=["Y"], name="unsup_0")
        initializers: dict[str, Any] = {}

        result = _execute_generation_ops(node, initializers)

        assert result is None


class TestExecuteAggregationOps:
    """Test _execute_aggregation_ops function."""

    def test_reduce_sum_with_axes(self):
        """Test ReduceSum with specified axes."""
        node = helper.make_node("ReduceSum", inputs=["X", "Axes"], outputs=["Y"], name="redsum_0")
        x_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        axes_array = np.array([1])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Axes": numpy_helper.from_array(axes_array, "Axes"),
        }
        node.attribute.append(helper.make_attribute("keepdims", 0))

        result = _execute_aggregation_ops(node, initializers, {})

        assert np.allclose(result, np.array([3.0, 7.0]))

    def test_reduce_sum_all_axes(self):
        """Test ReduceSum reducing all axes."""
        node = helper.make_node("ReduceSum", inputs=["X"], outputs=["Y"], name="redsum_0")
        x_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
        }
        node.attribute.append(helper.make_attribute("keepdims", 0))
        node.attribute.append(helper.make_attribute("noop_with_empty_axes", 0))

        result = _execute_aggregation_ops(node, initializers, {})

        assert np.allclose(result, np.array(10.0))

    def test_reduce_sum_noop(self):
        """Test ReduceSum with noop_with_empty_axes=True raises error."""
        node = helper.make_node("ReduceSum", inputs=["X"], outputs=["Y"], name="redsum_0")
        x_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
        }
        node.attribute.append(helper.make_attribute("noop_with_empty_axes", 1))

        # noop_with_empty_axes=1 is not supported
        with pytest.raises(ValueError, match="not supported"):
            _execute_aggregation_ops(node, initializers, {})

    def test_concatenates_along_axis(self):
        """Test Concat returns output shape from shapes dict."""
        node = helper.make_node("Concat", inputs=["X", "Y"], outputs=["Z"], name="concat_0", axis=0)
        x_array = np.array([[1.0, 2.0]])
        y_array = np.array([[3.0, 4.0]])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Y": numpy_helper.from_array(y_array, "Y"),
        }
        # When all inputs are in initializers, function returns shape from shapes dict
        shapes = {"Z": [2, 2]}

        result = _execute_aggregation_ops(node, initializers, shapes)

        # Result is the output shape when all inputs are in initializers
        assert np.array_equal(result, np.array([2, 2], dtype=np.int64))


class TestExecuteTypeAndLogicOps:
    """Test _execute_type_and_logic_ops function."""

    def test_cast_to_float32(self):
        """Test Cast to float32."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"], name="cast_0")
        node.attribute.append(helper.make_attribute("to", 1))  # float32
        int_array = np.array([1, 2, 3], dtype=np.int32)
        initializers = {
            "X": numpy_helper.from_array(int_array, "X"),
        }

        result = _execute_type_and_logic_ops(node, initializers, {})

        assert result.dtype == np.float32
        assert np.allclose(result, np.array([1.0, 2.0, 3.0]))

    def test_cast_unsupported_dtype(self):
        """Test Cast with unsupported dtype raises error."""
        node = helper.make_node("Cast", inputs=["X"], outputs=["Y"], name="cast_0")
        node.attribute.append(helper.make_attribute("to", 9999))  # Invalid dtype
        initializers = {
            "X": create_initializer("X", np.array([1.0, 2.0])),
        }

        with pytest.raises(ValueError, match="Unsupported Cast dtype"):
            _execute_type_and_logic_ops(node, initializers, {})

    def test_returns_elementwise_equality(self):
        """Test Equal returns True for matching elements."""
        node = helper.make_node("Equal", inputs=["X", "Y"], outputs=["Z"], name="eq_0")
        initializers = {
            "X": create_initializer("X", np.array([1.0, 2.0, 3.0])),
            "Y": create_initializer("Y", np.array([1.0, 3.0, 3.0])),
        }

        result = _execute_type_and_logic_ops(node, initializers, {})

        assert np.array_equal(result, np.array([True, False, True]))

    def test_selects_on_condition(self):
        """Test Where selects from X or Y based on condition."""
        node = helper.make_node("Where", inputs=["Cond", "X", "Y"], outputs=["Z"], name="where_0")
        cond_array = np.array([True, False, True])
        x_array = np.array([1.0, 2.0, 3.0])
        y_array = np.array([10.0, 20.0, 30.0])
        initializers = {
            "Cond": numpy_helper.from_array(cond_array, "Cond"),
            "X": numpy_helper.from_array(x_array, "X"),
            "Y": numpy_helper.from_array(y_array, "Y"),
        }

        result = _execute_type_and_logic_ops(node, initializers, {})

        expected = np.where(cond_array, x_array, y_array)
        assert np.allclose(result, expected)

    def test_broadcasts_to_target_shape(self):
        """Test Expand broadcasts input to target shape."""
        node = helper.make_node("Expand", inputs=["X"], outputs=["Y"], name="expand_0")
        initializers = {
            "X": create_initializer("X", np.array([[1.0, 2.0]])),
        }
        shapes = {"Y": [3, 2]}

        result = _execute_type_and_logic_ops(node, initializers, shapes)

        expected = np.broadcast_to(np.array([[1.0, 2.0]]), (3, 2))
        assert np.allclose(result, expected)


class TestShapeManipulationOps:
    """Test _execute_shape_manipulation_ops function."""

    def test_reshapes_to_target_dimensions(self):
        """Test Reshape reshapes input to target shape."""
        node = helper.make_node("Reshape", inputs=["X", "Shape"], outputs=["Y"], name="reshape_0")
        x_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        shape_array = np.array([2, 3])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Shape": numpy_helper.from_array(shape_array, "Shape"),
        }

        result = _execute_shape_manipulation_ops(node, initializers, {}, {})

        expected = x_array.reshape(2, 3)
        assert np.allclose(result, expected)

    def test_gathers_indexed_elements(self):
        """Test Gather selects elements at specified indices."""
        node = helper.make_node(
            "Gather", inputs=["X", "Indices"], outputs=["Y"], name="gather_0", axis=0
        )
        x_array = np.array([1.0, 2.0, 3.0, 4.0])
        indices_array = np.array([0, 2, 3])
        initializers = {
            "X": numpy_helper.from_array(x_array, "X"),
            "Indices": numpy_helper.from_array(indices_array, "Indices"),
        }
        nodes_dict = {"X": helper.make_node("Identity", inputs=["input"], outputs=["X"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        expected = np.array([1.0, 3.0, 4.0])
        assert np.allclose(result, expected)

    def test_slice_basic(self):
        """Test basic Slice operation."""
        node = helper.make_node(
            "Slice", inputs=["data", "starts", "ends"], outputs=["output"], name="slice_0"
        )
        data = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([4], dtype=np.int64)

        initializers = {
            "data": numpy_helper.from_array(data, "data"),
            "starts": numpy_helper.from_array(starts, "starts"),
            "ends": numpy_helper.from_array(ends, "ends"),
        }
        nodes_dict = {"data": helper.make_node("Identity", inputs=["input"], outputs=["data"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        expected = np.array([1, 2, 3], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_slice_with_axes(self):
        """Test Slice with explicit axes parameter."""
        node = helper.make_node(
            "Slice", inputs=["data", "starts", "ends", "axes"], outputs=["output"], name="slice_1"
        )
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([3], dtype=np.int64)
        axes = np.array([1], dtype=np.int64)

        initializers = {
            "data": numpy_helper.from_array(data, "data"),
            "starts": numpy_helper.from_array(starts, "starts"),
            "ends": numpy_helper.from_array(ends, "ends"),
            "axes": numpy_helper.from_array(axes, "axes"),
        }
        nodes_dict = {"data": helper.make_node("Identity", inputs=["input"], outputs=["data"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        expected = data[:, 1:3]
        assert np.allclose(result, expected)

    def test_slice_with_steps(self):
        """Test Slice with steps parameter."""
        node = helper.make_node(
            "Slice",
            inputs=["data", "starts", "ends", "axes", "steps"],
            outputs=["output"],
            name="slice_2",
        )
        data = np.arange(10, dtype=np.float32)
        starts = np.array([1], dtype=np.int64)
        ends = np.array([8], dtype=np.int64)
        axes = np.array([0], dtype=np.int64)
        steps = np.array([2], dtype=np.int64)

        initializers = {
            "data": numpy_helper.from_array(data, "data"),
            "starts": numpy_helper.from_array(starts, "starts"),
            "ends": numpy_helper.from_array(ends, "ends"),
            "axes": numpy_helper.from_array(axes, "axes"),
            "steps": numpy_helper.from_array(steps, "steps"),
        }
        nodes_dict = {"data": helper.make_node("Identity", inputs=["input"], outputs=["data"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        expected = data[1:8:2]
        assert np.allclose(result, expected)

    def test_unsqueeze_single_axis(self):
        """Test Unsqueeze operation with single axis."""
        node = helper.make_node(
            "Unsqueeze", inputs=["data", "axes"], outputs=["output"], name="unsqueeze_0"
        )
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        axes = np.array([0], dtype=np.int64)

        initializers = {
            "data": numpy_helper.from_array(data, "data"),
            "axes": numpy_helper.from_array(axes, "axes"),
        }
        nodes_dict = {"data": helper.make_node("Identity", inputs=["input"], outputs=["data"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        expected = np.expand_dims(data, axis=0)
        assert np.array_equal(result, expected)

    def test_unsqueeze_multiple_axes(self):
        """Test Unsqueeze with multiple axes."""
        node = helper.make_node(
            "Unsqueeze", inputs=["data", "axes"], outputs=["output"], name="unsqueeze_1"
        )
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        axes = np.array([0, 3], dtype=np.int64)

        initializers = {
            "data": numpy_helper.from_array(data, "data"),
            "axes": numpy_helper.from_array(axes, "axes"),
        }
        nodes_dict = {"data": helper.make_node("Identity", inputs=["input"], outputs=["data"])}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)

        assert result.shape == (1, 2, 2, 1)

    def test_scalar_shape_in_concat(self):
        """Test scalar shape conversion in Concat operation."""
        node = helper.make_node(
            "Concat", inputs=["shape1", "shape2"], outputs=["output"], name="concat_0", axis=0
        )
        initializers = {
            "shape1": numpy_helper.from_array(np.array([2], dtype=np.int64), "shape1"),
            "shape2": numpy_helper.from_array(np.array([3], dtype=np.int64), "shape2"),
        }
        # Simulate scalar shape in shapes dict
        shapes = {"output": 5}  # scalar shape that needs conversion

        result = _execute_concat(node, initializers, shapes)

        expected = np.array([5], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_scalar_shape_in_expand(self):
        """Test scalar shape conversion in Expand operation."""
        node = helper.make_node(
            "Expand", inputs=["data", "shape"], outputs=["output"], name="expand_0"
        )
        data = np.array(5.0, dtype=np.float32)
        initializers = {
            "data": numpy_helper.from_array(data, "data"),
        }
        # Simulate scalar shape in shapes dict
        shapes = {"output": 3}  # scalar shape that needs conversion

        result = _execute_type_and_logic_ops(node, initializers, shapes)

        expected = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        assert np.array_equal(result, expected)


class TestFuseConstantNodes:
    """Test _fuse_constant_nodes function."""

    def test_fuse_simple_add(self):
        """Test fusing a simple Add with constant inputs in initializers."""
        add_node = helper.make_node("Add", inputs=["C1", "C2"], outputs=["Result"], name="add_0")
        # Add a node that uses the result so it stays in initializers
        identity_node = helper.make_node(
            "Identity", inputs=["Result"], outputs=["Output"], name="identity_0"
        )

        nodes = [add_node, identity_node]
        # Provide constants in initializers so Add can be folded
        initializers = {
            "C1": create_initializer("C1", np.array([1.0, 2.0])),
            "C2": create_initializer("C2", np.array([1.0, 1.0])),
        }
        shapes: dict[str, int | list[int]] = {}

        result_nodes, result_initializers = _fuse_constant_nodes(nodes, initializers, shapes)

        # Add node should be removed, Identity should remain
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Identity"
        # Result should be in initializers since Identity uses it
        assert "Result" in result_initializers
        # Verify the computed result
        result_value = numpy_helper.to_array(result_initializers["Result"])
        assert np.allclose(result_value, np.array([2.0, 3.0]))

    def test_fuse_shape_node(self):
        """Test fusing a Shape node when followed by another operation."""
        shape_node = helper.make_node("Shape", inputs=["X"], outputs=["ShapeOut"], name="shape_0")
        # Add a node that uses the shape so it stays in initializers
        reshape_node = helper.make_node(
            "Reshape", inputs=["X", "ShapeOut"], outputs=["Y"], name="reshape_0"
        )

        nodes = [shape_node, reshape_node]
        initializers: dict[str, Any] = {}
        shapes = {"ShapeOut": [2, 3]}

        result_nodes, result_initializers = _fuse_constant_nodes(nodes, initializers, shapes)

        # Shape node should be removed, Reshape should remain
        assert len(result_nodes) == 1
        assert result_nodes[0].op_type == "Reshape"
        # ShapeOut should be in initializers since Reshape uses it
        assert "ShapeOut" in result_initializers
        shape_value = numpy_helper.to_array(result_initializers["ShapeOut"])
        assert np.array_equal(shape_value, np.array([2, 3], dtype=np.int64))

    def test_no_fuse_missing_inputs(self):
        """Test no folding when inputs are missing."""
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"], name="add_0")
        nodes = [add_node]
        initializers: dict[str, Any] = {}
        shapes: dict[str, int | list[int]] = {}

        result_nodes, _result_initializers = _fuse_constant_nodes(nodes, initializers, shapes)

        # Node should remain since inputs not available
        assert len(result_nodes) == 1
        assert add_node in result_nodes

    def test_no_fuse_unsupported_op(self):
        """Test no folding for unsupported operations."""
        unsupported_node = helper.make_node(
            "UnsupportedOp", inputs=["X"], outputs=["Y"], name="unsup_0"
        )
        nodes = [unsupported_node]
        initializers = {"X": create_initializer("X", np.array([1.0]))}
        shapes: dict[str, int | list[int]] = {}

        result_nodes, _result_initializers = _fuse_constant_nodes(nodes, initializers, shapes)

        # Node should remain since operation unsupported
        assert len(result_nodes) == 1
        assert unsupported_node in result_nodes
