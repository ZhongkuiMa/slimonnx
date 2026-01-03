"""Tests for constant folding operations."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
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
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestExecuteRange:
    """Test _execute_range function."""

    def test_execute_range_basic(self):
        """Test basic range generation."""
        start = create_initializer("start", np.array(0, dtype=np.int64))
        limit = create_initializer("limit", np.array(10, dtype=np.int64))
        delta = create_initializer("delta", np.array(1, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_range(node, initializers)
        assert result is not None
        assert np.array_equal(result, np.arange(0, 10, 1))

    def test_execute_range_with_step(self):
        """Test range with custom step."""
        start = create_initializer("start", np.array(0, dtype=np.int64))
        limit = create_initializer("limit", np.array(10, dtype=np.int64))
        delta = create_initializer("delta", np.array(2, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_range(node, initializers)
        assert result is not None
        assert np.array_equal(result, np.arange(0, 10, 2))

    def test_execute_range_missing_inputs(self):
        """Test range with missing inputs returns None."""
        start = create_initializer("start", np.array(0, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start}

        result = _execute_range(node, initializers)
        assert result is None

    def test_execute_range_float(self):
        """Test range with float values."""
        start = create_initializer("start", np.array(0.0, dtype=np.float32))
        limit = create_initializer("limit", np.array(5.0, dtype=np.float32))
        delta = create_initializer("delta", np.array(0.5, dtype=np.float32))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_range(node, initializers)
        assert result is not None
        assert len(result) > 0


class TestExecuteReduceSum:
    """Test _execute_reduce_sum function."""

    def test_execute_reduce_sum_with_axes(self):
        """Test ReduceSum with explicit axes."""
        data = create_initializer(
            "data",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        axes_array = np.array([1], dtype=np.int64)
        axes = onnx.numpy_helper.from_array(axes_array, "axes")

        node = helper.make_node("ReduceSum", inputs=["data", "axes"], outputs=["Y"], keepdims=1)
        initializers = {"data": data, "axes": axes}

        result = _execute_reduce_sum(node, initializers)
        assert result is not None
        assert result.shape == (2, 1)

    def test_execute_reduce_sum_without_axes(self):
        """Test ReduceSum without axes (reduce all)."""
        data = create_initializer(
            "data",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )

        node = helper.make_node("ReduceSum", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_reduce_sum(node, initializers)
        assert result is not None

    def test_execute_reduce_sum_keepdims_false(self):
        """Test ReduceSum with keepdims=False."""
        data = create_initializer(
            "data",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        axes_array = np.array([1], dtype=np.int64)
        axes = onnx.numpy_helper.from_array(axes_array, "axes")

        node = helper.make_node("ReduceSum", inputs=["data", "axes"], outputs=["Y"], keepdims=0)
        initializers = {"data": data, "axes": axes}

        result = _execute_reduce_sum(node, initializers)
        assert result is not None
        assert result.shape == (2,)


class TestExecuteBinaryArithmetic:
    """Test _execute_binary_arithmetic function."""

    def test_execute_add(self):
        """Test Add operation."""
        a = create_initializer("a", np.array([1, 2, 3], dtype=np.float32))
        b = create_initializer("b", np.array([4, 5, 6], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.array_equal(result, np.array([5, 7, 9], dtype=np.float32))

    def test_execute_sub(self):
        """Test Sub operation."""
        a = create_initializer("a", np.array([10, 20, 30], dtype=np.float32))
        b = create_initializer("b", np.array([1, 2, 3], dtype=np.float32))

        node = helper.make_node("Sub", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.array_equal(result, np.array([9, 18, 27], dtype=np.float32))

    def test_execute_mul(self):
        """Test Mul operation."""
        a = create_initializer("a", np.array([2, 3, 4], dtype=np.float32))
        b = create_initializer("b", np.array([5, 6, 7], dtype=np.float32))

        node = helper.make_node("Mul", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.array_equal(result, np.array([10, 18, 28], dtype=np.float32))

    def test_execute_div_float(self):
        """Test Div operation with floats."""
        a = create_initializer("a", np.array([10.0, 20.0, 30.0], dtype=np.float32))
        b = create_initializer("b", np.array([2.0, 4.0, 5.0], dtype=np.float32))

        node = helper.make_node("Div", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.allclose(result, np.array([5.0, 5.0, 6.0], dtype=np.float32))

    def test_execute_div_int(self):
        """Test Div operation with integers."""
        a = create_initializer("a", np.array([10, 20, 30], dtype=np.int32))
        b = create_initializer("b", np.array([2, 5, 10], dtype=np.int32))

        node = helper.make_node("Div", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        # Division should produce expected results
        assert result is not None
        assert len(result) == 3

    def test_execute_matmul(self):
        """Test MatMul operation."""
        a = create_initializer("a", np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = create_initializer("b", np.array([[5, 6], [7, 8]], dtype=np.float32))

        node = helper.make_node("MatMul", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        expected = np.matmul(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        assert np.array_equal(result, expected)

    def test_execute_pow(self):
        """Test Pow operation."""
        a = create_initializer("a", np.array([2, 3, 4], dtype=np.float32))
        b = create_initializer("b", np.array([2, 3, 2], dtype=np.float32))

        node = helper.make_node("Pow", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_binary_arithmetic(node, initializers)
        assert np.allclose(result, np.array([4, 27, 16], dtype=np.float32))


class TestCanFoldNode:
    """Test _can_fold_node function."""

    def test_can_fold_node_all_in_initializers(self):
        """Test node can be folded when all inputs in initializers."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))
        b = create_initializer("b", np.array([3, 4], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _can_fold_node(node, initializers, [])
        assert result is True

    def test_can_fold_node_partial_in_nodes_to_delete(self):
        """Test node can be folded when inputs in initializers or nodes_to_delete."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "Y1"], outputs=["Y"])
        initializers = {"a": a}
        nodes_to_delete = ["Y1"]

        result = _can_fold_node(node, initializers, nodes_to_delete)
        assert result is True

    def test_can_fold_node_missing_input(self):
        """Test node cannot be folded when inputs are missing."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a}

        result = _can_fold_node(node, initializers, [])
        assert result is False


class TestExecuteElementwiseOps:
    """Test _execute_elementwise_ops function."""

    def test_execute_relu(self):
        """Test Relu operation."""
        data = create_initializer("data", np.array([-1, 0, 1, 2], dtype=np.float32))

        node = helper.make_node("Relu", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_elementwise_ops(node, initializers)
        assert result is not None
        assert np.array_equal(result, np.array([0, 0, 1, 2], dtype=np.float32))

    def test_execute_neg(self):
        """Test Neg operation."""
        data = create_initializer("data", np.array([1, 2, -3], dtype=np.float32))

        node = helper.make_node("Neg", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_elementwise_ops(node, initializers)
        assert result is not None
        assert np.array_equal(result, np.array([-1, -2, 3], dtype=np.float32))

    def test_execute_elementwise_add(self):
        """Test elementwise Add via _execute_elementwise_ops."""
        a = create_initializer("a", np.array([1, 2], dtype=np.float32))
        b = create_initializer("b", np.array([3, 4], dtype=np.float32))

        node = helper.make_node("Add", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_elementwise_ops(node, initializers)
        assert result is not None
        assert np.array_equal(result, np.array([4, 6], dtype=np.float32))

    def test_execute_unknown_elementwise_op(self):
        """Test unknown elementwise operation returns None."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("UnknownOp", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_elementwise_ops(node, initializers)
        assert result is None


class TestExecuteTypeAndLogicOps:
    """Test _execute_type_and_logic_ops function."""

    def test_execute_cast_to_float(self):
        """Test Cast to float."""
        data = create_initializer("data", np.array([1, 2, 3], dtype=np.int32))

        node = helper.make_node("Cast", inputs=["data"], outputs=["Y"], to=TensorProto.FLOAT)
        initializers = {"data": data}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result is not None
        assert result.dtype == np.float32

    def test_execute_equal(self):
        """Test Equal operation."""
        a = create_initializer("a", np.array([1, 2, 3], dtype=np.int32))
        b = create_initializer("b", np.array([1, 2, 4], dtype=np.int32))

        node = helper.make_node("Equal", inputs=["a", "b"], outputs=["Y"])
        initializers = {"a": a, "b": b}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result is not None
        assert np.array_equal(result, np.array([True, True, False]))

    def test_execute_where(self):
        """Test Where operation."""
        cond = create_initializer("cond", np.array([True, False, True], dtype=bool))
        x = create_initializer("x", np.array([1, 2, 3], dtype=np.float32))
        y = create_initializer("y", np.array([10, 20, 30], dtype=np.float32))

        node = helper.make_node("Where", inputs=["cond", "x", "y"], outputs=["Y"])
        initializers = {"cond": cond, "x": x, "y": y}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result is not None
        assert np.array_equal(result, np.array([1, 20, 3], dtype=np.float32))

    def test_execute_expand(self):
        """Test Expand operation."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32).reshape(2, 1))

        node = helper.make_node("Expand", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}
        shapes = {"Y": (2, 3)}

        result = _execute_type_and_logic_ops(node, initializers, shapes)
        assert result is not None
        assert result.shape == (2, 3)

    def test_execute_unknown_logic_op(self):
        """Test unknown logic operation returns None."""
        data = create_initializer("data", np.array([1], dtype=np.float32))

        node = helper.make_node("UnknownLogic", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_type_and_logic_ops(node, initializers, {})
        assert result is None


class TestExecuteShapeManipulationOps:
    """Test _execute_shape_manipulation_ops function."""

    def test_execute_reshape(self):
        """Test Reshape operation."""
        data = create_initializer("data", np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
        shape_array = np.array([2, 3], dtype=np.int64)
        shape_init = onnx.numpy_helper.from_array(shape_array, "shape")

        node = helper.make_node("Reshape", inputs=["data", "shape"], outputs=["Y"])
        initializers = {"data": data, "shape": shape_init}
        nodes_dict: dict[str, Any] = {}

        result = _execute_shape_manipulation_ops(node, initializers, {}, nodes_dict)
        assert result is not None
        assert result.shape == (2, 3)

    def test_execute_unknown_shape_op(self):
        """Test unknown shape manipulation operation returns None."""
        data = create_initializer("data", np.array([1, 2, 3], dtype=np.float32))

        node = helper.make_node("UnknownShape", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_shape_manipulation_ops(node, initializers, {}, {})
        assert result is None


class TestExecuteAggregationOps:
    """Test _execute_aggregation_ops function."""

    def test_execute_concat_with_shape(self):
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
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_execute_unknown_aggregation_op(self):
        """Test unknown aggregation operation returns None."""
        data = create_initializer("data", np.array([1, 2], dtype=np.float32))

        node = helper.make_node("UnknownAgg", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_aggregation_ops(node, initializers, {})
        assert result is None


class TestExecuteGenerationOps:
    """Test _execute_generation_ops function."""

    def test_execute_range_via_generation(self):
        """Test Range via _execute_generation_ops."""
        start = create_initializer("start", np.array(0, dtype=np.int64))
        limit = create_initializer("limit", np.array(5, dtype=np.int64))
        delta = create_initializer("delta", np.array(1, dtype=np.int64))

        node = helper.make_node("Range", inputs=["start", "limit", "delta"], outputs=["Y"])
        initializers = {"start": start, "limit": limit, "delta": delta}

        result = _execute_generation_ops(node, initializers)
        assert result is not None
        assert len(result) == 5

    def test_execute_unknown_generation_op(self):
        """Test unknown generation operation returns None."""
        data = create_initializer("data", np.array([1], dtype=np.float32))

        node = helper.make_node("UnknownGen", inputs=["data"], outputs=["Y"])
        initializers = {"data": data}

        result = _execute_generation_ops(node, initializers)
        assert result is None


class TestFuseConstantNodes:
    """Test _fuse_constant_nodes function."""

    def test_fuse_constant_nodes_shape(self):
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

    def test_fuse_constant_nodes_arithmetic(self):
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

    def test_fuse_constant_nodes_no_foldable_nodes(self):
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
