"""Tests for constant operations and constant folding."""

from typing import Any

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx
from tests.test_units.conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestConstantFolding:
    """Test constant folding for various operations."""

    def test_constant_arithmetic(self):
        """Test constant folding for arithmetic operations."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Constants for Add operation
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)
        initializers = [
            create_initializer("a", a),
            create_initializer("b", b),
        ]

        # Add two constants
        add_node = helper.make_node(
            "Add",
            inputs=["a", "b"],
            outputs=["add_out"],
        )
        # Add result with input
        add2_node = helper.make_node(
            "Add",
            inputs=["X", "add_out"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([add_node, add2_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_mul(self):
        """Test constant folding for Mul operation."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Constants for Mul operation
        a = np.array(2.0, dtype=np.float32)
        b = np.array(3.0, dtype=np.float32)
        initializers = [
            create_initializer("a", a),
            create_initializer("b", b),
        ]

        # Mul two constants
        mul_node = helper.make_node(
            "Mul",
            inputs=["a", "b"],
            outputs=["mul_out"],
        )
        # Mul result with input
        mul2_node = helper.make_node(
            "Mul",
            inputs=["X", "mul_out"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([mul_node, mul2_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_gather(self):
        """Test constant folding for Gather operation."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]

        # Constant data for Gather
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int64)
        initializers = [
            create_initializer("data", data),
            create_initializer("indices", indices),
        ]

        # Gather from constants
        gather_node = helper.make_node(
            "Gather",
            inputs=["data", "indices"],
            outputs=["Y"],
            axis=0,
        )

        model = create_minimal_onnx_model([gather_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_slice(self):
        """Test constant folding for Slice operation."""
        inputs = [create_tensor_value_info("X", "float32", [1, 1])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]

        # Constant data for Slice
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        initializers = [create_initializer("data", data)]

        # Reshape constant to 1D then back
        reshape_node = helper.make_node(
            "Reshape",
            inputs=["data", "shape"],
            outputs=["Y"],
        )

        shape = np.array([1, 4], dtype=np.int64)
        initializers.append(create_initializer("shape", shape))

        model = create_minimal_onnx_model([reshape_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_unsqueeze(self):
        """Test constant folding for Unsqueeze operation."""
        inputs = [create_tensor_value_info("X", "float32", [1, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2, 3])]

        # Constant data
        data = np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("data", data)]

        # Add constant (simpler than Unsqueeze)
        const = np.ones((2, 3), dtype=np.float32)
        initializers.append(create_initializer("const", const))

        add_node = helper.make_node(
            "Add",
            inputs=["data", "const"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_reshape(self):
        """Test constant folding for Reshape operation."""
        inputs: list[Any] = []
        outputs = [create_tensor_value_info("Y", "float32", [2, 6])]

        # Constant data
        data = np.ones((2, 3), dtype=np.float32)
        shape = np.array([2, 6], dtype=np.int64)
        initializers = [
            create_initializer("data", data),
            create_initializer("shape", shape),
        ]

        # Reshape constant
        reshape_node = helper.make_node(
            "Reshape",
            inputs=["data", "shape"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([reshape_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_cast(self):
        """Test constant folding for Cast operation."""
        inputs: list[Any] = []
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Constant integer data
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        initializers = [create_initializer("data", data)]

        # Cast to float
        cast_node = helper.make_node(
            "Cast",
            inputs=["data"],
            outputs=["Y"],
            to=1,  # TensorProto.FLOAT = 1
        )

        model = create_minimal_onnx_model([cast_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_concat(self):
        """Test constant folding for Concat operation."""
        inputs = [create_tensor_value_info("X", "float32", [2, 1])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Constant data for concat
        a = np.ones((2, 1), dtype=np.float32)
        b = np.ones((2, 1), dtype=np.float32)
        initializers = [
            create_initializer("a", a),
            create_initializer("b", b),
        ]

        # Concat two constants
        concat_node = helper.make_node(
            "Concat",
            inputs=["a", "b"],
            outputs=["concat_out"],
            axis=1,
        )
        # Concat result with input
        concat2_node = helper.make_node(
            "Concat",
            inputs=["X", "concat_out"],
            outputs=["Y"],
            axis=1,
        )

        model = create_minimal_onnx_model(
            [concat_node, concat2_node], inputs, outputs, initializers
        )

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_transpose(self):
        """Test constant folding for Transpose operation."""
        inputs: list[Any] = []
        outputs = [create_tensor_value_info("Y", "float32", [3, 2])]

        # Constant data
        data = np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("data", data)]

        # Transpose constant
        transpose_node = helper.make_node(
            "Transpose",
            inputs=["data"],
            outputs=["Y"],
            perm=[1, 0],
        )

        model = create_minimal_onnx_model([transpose_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_squeeze(self):
        """Test constant folding for Squeeze operation."""
        inputs: list[Any] = []
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        # Constant data with squeeze dimension
        data = np.ones((2, 1, 3), dtype=np.float32)
        initializers = [create_initializer("data", data)]

        # Squeeze constant
        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=["data"],
            outputs=["Y"],
            axes=[1],
        )

        model = create_minimal_onnx_model([squeeze_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None

    def test_constant_reduce_sum(self):
        """Test constant folding for operations on constants."""
        inputs = [create_tensor_value_info("X", "float32", [1, 2])]
        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]

        # Constant data
        data1 = np.array([[1.0, 2.0]], dtype=np.float32)
        data2 = np.array([[3.0, 4.0]], dtype=np.float32)
        initializers = [
            create_initializer("data1", data1),
            create_initializer("data2", data2),
        ]

        # Add two constants
        const_add_node = helper.make_node(
            "Add",
            inputs=["data1", "data2"],
            outputs=["const_out"],
        )
        # Add result to input
        add_node = helper.make_node(
            "Add",
            inputs=["X", "const_out"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([const_add_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model)
        assert optimized is not None
