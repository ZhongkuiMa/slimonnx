"""Tests for constant folding operations."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from slimonnx.optimize_onnx._cst_op import (
    _can_fold_node,
    _execute_aggregation_ops,
    _execute_binary_arithmetic,
    _execute_elementwise_ops,
    _execute_generation_ops,
    _execute_range,
    _execute_reduce_sum,
    _execute_shape_manipulation_ops,
    _execute_type_and_logic_ops,
    _fuse_constant_nodes,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestExecuteRange:
    """Test _execute_range function."""

    @pytest.mark.parametrize(
        ("start_val", "limit_val", "delta_val", "dtype", "expected"),
        [
            pytest.param(0, 10, 1, np.int64, np.arange(0, 10, 1), id="basic_int"),
            pytest.param(0, 10, 2, np.int64, np.arange(0, 10, 2), id="custom_step"),
            pytest.param(0.0, 5.0, 0.5, np.float32, np.arange(0.0, 5.0, 0.5), id="float_range"),
        ],
    )
    def test_with_inputs(self, start_val, limit_val, delta_val, dtype, expected):
        """Test range generation with various inputs."""
        start = create_initializer("start", np.array(start_val, dtype=dtype))
        limit = create_initializer("limit", np.array(limit_val, dtype=dtype))
        delta = create_initializer("delta", np.array(delta_val, dtype=dtype))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_range(node, initializers)
        if dtype == np.float32:
            assert np.allclose(result, expected)
        else:
            assert np.array_equal(result, expected)

    def test_missing_inputs(self):
        """Test range with missing inputs returns None."""
        start = create_initializer("start", np.array(0, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start}

        result = _execute_range(node, initializers)
        assert result is None


class TestExecuteReduceSum:
    """Test _execute_reduce_sum function."""

    @pytest.mark.parametrize(
        ("with_axes", "keepdims", "expected_shape"),
        [
            (True, True, (2, 1)),
            (True, False, (2,)),
            (False, True, ()),
        ],
        ids=["axes_keepdims_true", "axes_keepdims_false", "no_axes"],
    )
    def test_executes_with_axis_options(self, with_axes, keepdims, expected_shape):
        """Test ReduceSum with various axis and keepdims combinations."""
        data = create_initializer(
            "data",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )

        if with_axes:
            axes_array = np.array([1], dtype=np.int64)
            axes = onnx.numpy_helper.from_array(axes_array, "axes")
            node = helper.make_node(
                "ReduceSum", inputs=["data", "axes"], outputs=["Y"], keepdims=int(keepdims)
            )
            initializers = {"data": data, "axes": axes}
        else:
            node = helper.make_node("ReduceSum", inputs=["data"], outputs=["Y"])
            initializers = {"data": data}

        result = _execute_reduce_sum(node, initializers)
        assert isinstance(result, np.ndarray)
        if expected_shape == ():
            assert np.isclose(result, 21.0)
        else:
            assert result.shape == expected_shape


class TestExecuteBinaryArithmetic:
    """Test _execute_binary_arithmetic function."""

    @pytest.mark.parametrize(
        ("op", "a_val", "b_val", "expected"),
        [
            pytest.param(
                "Add",
                np.array([1, 2, 3], dtype=np.float32),
                np.array([4, 5, 6], dtype=np.float32),
                np.array([5, 7, 9], dtype=np.float32),
                id="add",
            ),
            pytest.param(
                "Sub",
                np.array([10, 20, 30], dtype=np.float32),
                np.array([1, 2, 3], dtype=np.float32),
                np.array([9, 18, 27], dtype=np.float32),
                id="sub",
            ),
            pytest.param(
                "Mul",
                np.array([2, 3, 4], dtype=np.float32),
                np.array([5, 6, 7], dtype=np.float32),
                np.array([10, 18, 28], dtype=np.float32),
                id="mul",
            ),
        ],
    )
    def test_float_operations(self, op, a_val, b_val, expected):
        """Test binary arithmetic operations with floats."""
        a = create_initializer("a", a_val)
        b = create_initializer("b", b_val)

        node = helper.make_node(op, inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype_name",
        [
            "float",
            "int",
        ],
        ids=["float_division", "int_division"],
    )
    def test_divides_operands(self, dtype_name):
        """Test Div operation with different numeric types."""
        if dtype_name == "float":
            a = create_initializer("a", np.array([10.0, 20.0, 30.0], dtype=np.float32))
            b = create_initializer("b", np.array([2.0, 4.0, 5.0], dtype=np.float32))
            expected = np.array([5.0, 5.0, 6.0], dtype=np.float32)
        else:
            a = create_initializer("a", np.array([10, 20, 30], dtype=np.int32))
            b = create_initializer("b", np.array([2, 5, 10], dtype=np.int32))
            expected = np.array([5, 4, 3], dtype=np.int32)

        node = helper.make_node("Div", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        if dtype_name == "float":
            assert np.allclose(result, expected)
        else:
            assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        ("op_name", "a_val", "b_val", "expected_val"),
        [
            (
                "MatMul",
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[5, 6], [7, 8]], dtype=np.float32),
                np.matmul(
                    np.array([[1, 2], [3, 4]]),
                    np.array([[5, 6], [7, 8]]),
                ),
            ),
            (
                "Pow",
                np.array([2, 3, 4], dtype=np.float32),
                np.array([2, 3, 2], dtype=np.float32),
                np.array([4, 27, 16], dtype=np.float32),
            ),
        ],
        ids=["matmul_2d", "elementwise_pow"],
    )
    def test_executes_higher_order_operations(self, op_name, a_val, b_val, expected_val):
        """Test MatMul and Pow operations."""
        a = create_initializer("a", a_val)
        b = create_initializer("b", b_val)

        node = helper.make_node(op_name, inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.allclose(result, expected_val)


class TestCanFoldNode:
    """Test _can_fold_node function."""

    @pytest.mark.parametrize(
        ("all_in_initializers", "some_in_delete_list"),
        [
            (True, False),
            (False, True),
        ],
        ids=["all_constant", "mixed_constant_and_deleted"],
    )
    def test_determines_foldability(self, all_in_initializers, some_in_delete_list):
        """Test node foldability with various input source combinations."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))

        if all_in_initializers:
            b = create_initializer("b", np.array([3, 4], dtype=np.float32))
            node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
            initializers = {"a": a, "b": b}
            nodes_to_delete = []
        else:
            node = helper.make_node("Add", inputs=["a", "Y1"], outputs=["Y"])
            initializers = {"a": a}
            nodes_to_delete = ["Y1"]

        result = _can_fold_node(node, initializers, nodes_to_delete)
        assert result is True

    def test_missing_input(self):
        """Test node cannot be folded when inputs are missing."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a}

        result = _can_fold_node(node, initializers, [])
        assert result is False


class TestExecuteElementwiseOps:
    """Test _execute_elementwise_ops function."""

    @pytest.mark.parametrize(
        ("op_name", "input_val", "expected_val"),
        [
            (
                "Relu",
                np.array([-1, 0, 1, 2], dtype=np.float32),
                np.array([0, 0, 1, 2], dtype=np.float32),
            ),
            (
                "Neg",
                np.array([1, 2, -3], dtype=np.float32),
                np.array([-1, -2, 3], dtype=np.float32),
            ),
        ],
        ids=["clips_to_zero", "negate"],
    )
    def test_executes_unary_ops(self, op_name, input_val, expected_val):
        """Test unary elementwise operations."""
        data = create_initializer("data", input_val)

        node = helper.make_node(op_name, inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_elementwise_ops(node, initializers)
        assert np.array_equal(result, expected_val)

    def test_elementwise_add(self):
        """Test elementwise Add via _execute_elementwise_ops."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))
        b = create_initializer("b", np.array([3, 4], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_elementwise_ops(node, initializers)
        assert np.array_equal(result, np.array([4, 6], dtype=np.float32))

    def test_unknown_op(self):
        """Test unknown elementwise operation returns None."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("UnknownOp", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_elementwise_ops(node, initializers)
        assert result is None


class TestExecuteTypeAndLogicOps:
    """Test _execute_type_and_logic_ops function."""

    def test_cast_to_float(self):
        """Test Cast to float."""
        data = create_initializer("data", np.array([1, 2, 3], dtype=np.int32))

        node = helper.make_node("Cast", inputs=["data"], outputs=["Y"], to=TensorProto.FLOAT)
        initializers = {"data": data}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result.dtype == np.float32

    def test_produces_elementwise_equality_mask(self):
        """Test Equal operation."""
        a = create_initializer("a", np.array([1, 2, 3], dtype=np.int32))
        b = create_initializer("b", np.array([1, 2, 4], dtype=np.int32))

        node = helper.make_node("Equal", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert np.array_equal(result, np.array([True, True, False]))

    @pytest.mark.parametrize(
        ("cond_val", "expected_val"),
        [
            (np.array([True, False, True], dtype=bool), np.array([1, 20, 3], dtype=np.float32)),
            (np.array([False, True, False], dtype=bool), np.array([10, 2, 30], dtype=np.float32)),
        ],
        ids=["standard_where", "inverted_condition"],
    )
    def test_selects_via_condition(self, cond_val, expected_val):
        """Test Where operation with different conditions."""
        cond = create_initializer("cond", cond_val)
        x = create_initializer("x", np.array([1, 2, 3], dtype=np.float32))
        y = create_initializer("y", np.array([10, 20, 30], dtype=np.float32))

        node = helper.make_node("Where", inputs=["cond", "x", "y"], outputs=["Y"])
        initializers = {"cond": cond, "x": x, "y": y}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert np.array_equal(result, expected_val)

    def test_broadcasts_to_target_shape(self):
        """Test Expand operation."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32).reshape(2, 1))

        node = helper.make_node("Expand", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}
        shapes = {"Y": (2, 3)}

        result = _execute_type_and_logic_ops(node, initializers, shapes)
        assert result.shape == (2, 3)

    def test_unknown_logic_op(self):
        """Test unknown logic operation returns None."""
        data = create_initializer("data", np.array([1], dtype=np.float32))

        node = helper.make_node("UnknownLogic", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result is None


class TestExecuteShapeManipulationOps:
    """Test _execute_shape_manipulation_ops function."""

    def test_produces_reshaped_array(self):
        """Test Reshape operation."""
        data = create_initializer("data", np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
        shape_array = np.array([2, 3], dtype=np.int64)
        shape_init = onnx.numpy_helper.from_array(shape_array, "shape")

        node = helper.make_node("Reshape", inputs=["data", "shape"], outputs=["Y"])
        initializers = {"data": data, "shape": shape_init}
        nodes_dict: dict[str, Any] = {}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)
        assert result.shape == (2, 3)

    def test_unknown_shape_op(self):
        """Test unknown shape manipulation operation returns None."""
        data = create_initializer("data", np.array([1, 2, 3], dtype=np.float32))

        node = helper.make_node("UnknownShape", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_shape_manipulation_ops(node, initializers, {}, {})
        assert result is None


class TestExecuteAggregationOps:
    """Test _execute_aggregation_ops function."""

    def test_concat_with_shape(self):
        """Test Concat when inputs are shape values."""
        a_array = np.array([2, 3], dtype=np.int64)
        b_array = np.array([4, 5], dtype=np.int64)
        a = onnx.numpy_helper.from_array(a_array, "a")
        b = onnx.numpy_helper.from_array(b_array, "b")

        node = helper.make_node("Concat", inputs=["a", "b"], outputs=["Y"], axis=0)
        initializers = {"a": a, "b": b}
        # This simulates concatenating shape values
        shapes = {"Y": [2, 3, 4, 5]}

        result = _execute_aggregation_ops(node, initializers, shapes)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([2, 3, 4, 5], dtype=np.int64))

    def test_unknown_aggregation_op(self):
        """Test unknown aggregation operation returns None."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("UnknownAgg", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_aggregation_ops(node, initializers, {})
        assert result is None


class TestExecuteGenerationOps:
    """Test _execute_generation_ops function."""

    def test_range_via_generation(self):
        """Test Range via _execute_generation_ops."""
        start = create_initializer("start", np.array(0, dtype=np.int64))
        limit = create_initializer("limit", np.array(5, dtype=np.int64))
        delta = create_initializer("delta", np.array(1, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_generation_ops(node, initializers)
        assert np.array_equal(result, np.arange(0, 5, 1))

    def test_unknown_generation_op(self):
        """Test unknown generation operation returns None."""
        data = create_initializer("data", np.array([1], dtype=np.float32))

        node = helper.make_node("UnknownGen", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_generation_ops(node, initializers)
        assert result is None


class TestFuseConstantNodes:
    """Test _fuse_constant_nodes function."""

    def test_removes_shape_node_when_foldable(self):
        """Test fusing Shape node to constant."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "int64", [2])

        shape_node = helper.make_node("Shape", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([shape_node], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        shapes = {"Y": [2, 3]}

        new_nodes, _ = _fuse_constant_nodes(nodes, initializers, shapes)
        # Shape node should be removed
        assert all(node.op_type != "Shape" for node in new_nodes)

    def test_removes_arithmetic_node_on_constant_inputs(self):
        """Test fusing arithmetic on constants."""
        a = create_initializer("a", np.array([1, 2, 3], dtype=np.float32))
        b = create_initializer("b", np.array([4, 5, 6], dtype=np.float32))

        X = create_tensor_value_info("X", "float32", [3])
        Y = create_tensor_value_info("Y", "float32", [3])

        add_node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])

        model = create_minimal_onnx_model([add_node], [X], [Y], [a, b])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        shapes = {"Y": [3]}

        new_nodes, _ = _fuse_constant_nodes(nodes, initializers, shapes)
        # Add node should be removed since inputs are constants
        assert all(node.op_type != "Add" for node in new_nodes)

    def test_no_foldable_nodes(self):
        """Test with no constant operations to fold."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu_node], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        shapes = {"Y": [2, 3]}

        new_nodes, _ = _fuse_constant_nodes(nodes, initializers, shapes)
        # Relu node should remain
        assert len(new_nodes) == 1
        assert new_nodes[0].op_type == "Relu"
