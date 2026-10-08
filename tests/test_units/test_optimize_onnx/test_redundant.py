"""Unit tests for redundant operation removal."""

__docformat__ = "restructuredtext"

import numpy as np
import onnxruntime as ort
import pytest
from _helpers import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)
from onnx import helper

from slimonnx.optimize_onnx import optimize_onnx


class TestRedundantOperations:
    """Test optimization on models with redundant patterns."""

    @pytest.mark.parametrize(
        ("op_type", "operand"),
        [
            pytest.param("Add", np.ones((1, 3), dtype=np.float32), id="add"),
            pytest.param("Mul", 2.0 * np.ones((1, 3), dtype=np.float32), id="mul"),
        ],
    )
    def test_optimizes_single_operations(self, op_type, operand):
        """Optimize model with basic arithmetic operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        operand_name = "operand"
        initializers = [create_initializer(operand_name, operand)]

        node = helper.make_node(op_type, inputs=["X", operand_name], outputs=["Y"])
        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

    def test_removes_chained_operations(self):
        """Optimize model with chained operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        b1 = np.ones((1, 3), dtype=np.float32)
        b2 = 2.0 * np.ones((1, 3), dtype=np.float32)
        initializers = [
            create_initializer("b1", b1),
            create_initializer("b2", b2),
        ]

        add_node = helper.make_node("Add", inputs=["X", "b1"], outputs=["temp"])
        mul_node = helper.make_node("Mul", inputs=["temp", "b2"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node, mul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

    def test_matmul_then_add_fuses(self):
        """MatMul followed by Add should fuse into Gemm - tests fusion across operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.eye(3, 2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["temp"])
        add_node = helper.make_node("Add", inputs=["temp", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)
        # MatMul+Add fusion produces Gemm
        gemm_nodes = [n for n in optimized.graph.node if n.op_type == "Gemm"]
        assert len(gemm_nodes) > 0

    def test_transpose_operations_handled(self):
        """Transpose operations should be handled correctly - tests transpose logic."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W = np.eye(2, 3, dtype=np.float32).T
        initializers = [create_initializer("W", W)]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

    def test_add_nonzero_value_kept(self):
        """X + non_zero value should be kept - tests non-zero threshold."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Small but non-zero value
        val = np.array([0.001, 0.001, 0.001], dtype=np.float32)
        initializers = [create_initializer("val", val)]

        add_node = helper.make_node("Add", inputs=["X", "val"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized
        # Non-zero Add should be kept
        add_nodes = [n for n in optimized.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 1

    @pytest.mark.parametrize(
        ("op_type", "identity", "initializer_first", "removed"),
        [
            ("Add", 0.0, False, True),
            ("Sub", 0.0, False, True),
            ("Mul", 1.0, False, True),
            ("Div", 1.0, False, True),
            ("Add", 0.0, True, True),
            ("Sub", 0.0, True, False),
            ("Mul", 1.0, True, True),
            ("Div", 1.0, True, False),
        ],
    )
    def test_redundant_arithmetic_respects_operand_order(
        self,
        op_type,
        identity,
        initializer_first,
        removed,
    ):
        """Identity removal preserves values for both operand orders."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3])
        value = np.full((1, 3), identity, dtype=np.float32)
        initializer = create_initializer("identity", value)
        arithmetic_inputs = ["identity", "X"] if initializer_first else ["X", "identity"]
        arithmetic = helper.make_node(op_type, inputs=arithmetic_inputs, outputs=["temp"])
        consumer = helper.make_node("Identity", inputs=["temp"], outputs=["Y"])
        model = create_minimal_onnx_model([arithmetic, consumer], [X], [Y], [initializer])

        optimized = optimize_onnx(
            model,
            remove_redundant_operations=True,
            has_batch_dim=True,
        )

        input_value = np.array([[2.0, 4.0, -2.0]], dtype=np.float32)
        original_out = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": input_value})[0]
        optimized_out = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": input_value})[0]
        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)
        assert any(node.op_type == op_type for node in optimized.graph.node) is not removed

    def test_redundant_public_output_preserves_name(self):
        """A public identity result keeps both its producer and value name."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3])
        zero = create_initializer("zero", np.zeros((1, 3), dtype=np.float32))
        add = helper.make_node("Add", inputs=["X", "zero"], outputs=["Y"])
        model = create_minimal_onnx_model([add], [X], [Y], [zero])

        optimized = optimize_onnx(
            model,
            remove_redundant_operations=True,
            has_batch_dim=True,
        )

        assert optimized.graph.output[0].name == "Y"
        assert [node.op_type for node in optimized.graph.node] == ["Add"]
        output = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": np.ones((1, 3), dtype=np.float32)})[0]
        np.testing.assert_array_equal(output, np.ones((1, 3), dtype=np.float32))

    def test_shared_identity_initializer_remains_live(self):
        """Removing one identity user cannot orphan a retained consumer."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3])
        zero = create_initializer("zero", np.zeros((1, 3), dtype=np.float32))
        unused_identity = helper.make_node("Add", inputs=["X", "zero"], outputs=["unused"])
        subtract = helper.make_node("Sub", inputs=["zero", "X"], outputs=["Y"])
        model = create_minimal_onnx_model([unused_identity, subtract], [X], [Y], [zero])

        optimized = optimize_onnx(
            model,
            remove_redundant_operations=True,
            has_batch_dim=True,
        )

        assert [node.op_type for node in optimized.graph.node] == ["Sub"]
        assert [initializer.name for initializer in optimized.graph.initializer] == ["zero"]
        input_value = np.array([[2.0, 4.0, -2.0]], dtype=np.float32)
        output = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": input_value})[0]
        np.testing.assert_array_equal(output, -input_value)

    def test_reshape_copy_zero_preserves_intermediate_shape(self):
        """A copied dimension is relative to the immediate Reshape input."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [3, 2])
        first_shape = create_initializer(
            "first_shape", np.array([3, 2], dtype=np.int64), dtype="int64"
        )
        second_shape = create_initializer(
            "second_shape", np.array([0, 2], dtype=np.int64), dtype="int64"
        )
        identity = helper.make_node("Identity", inputs=["X"], outputs=["source"])
        first = helper.make_node(
            "Reshape",
            inputs=["source", "first_shape"],
            outputs=["middle"],
        )
        second = helper.make_node(
            "Reshape",
            inputs=["middle", "second_shape"],
            outputs=["Y"],
        )
        model = create_minimal_onnx_model(
            [identity, first, second],
            [X],
            [Y],
            [first_shape, second_shape],
        )

        optimized = optimize_onnx(
            model,
            remove_redundant_operations=True,
            has_batch_dim=False,
        )

        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        output = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": input_value})[0]
        assert output.shape == (3, 2)
        np.testing.assert_array_equal(output, input_value.reshape(3, 2))

    def test_branched_reshape_chain_keeps_shared_producer(self):
        """Collapsing one consumer cannot orphan another reshape consumer."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [6])
        Z = create_tensor_value_info("Z", "float32", [3, 2])
        first_shape = create_initializer(
            "first_shape", np.array([3, 2], dtype=np.int64), dtype="int64"
        )
        second_shape = create_initializer(
            "second_shape", np.array([6], dtype=np.int64), dtype="int64"
        )
        identity = helper.make_node("Identity", inputs=["X"], outputs=["source"])
        first = helper.make_node(
            "Reshape",
            inputs=["source", "first_shape"],
            outputs=["middle"],
        )
        second = helper.make_node(
            "Reshape",
            inputs=["middle", "second_shape"],
            outputs=["Y"],
        )
        branch = helper.make_node("Identity", inputs=["middle"], outputs=["Z"])
        model = create_minimal_onnx_model(
            [identity, first, second, branch],
            [X],
            [Y, Z],
            [first_shape, second_shape],
        )

        optimized = optimize_onnx(
            model,
            remove_redundant_operations=True,
            has_batch_dim=False,
        )

        input_value = np.arange(6, dtype=np.float32).reshape(2, 3)
        outputs = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"X": input_value})
        np.testing.assert_array_equal(outputs[0], input_value.reshape(6))
        np.testing.assert_array_equal(outputs[1], input_value.reshape(3, 2))

    def test_add_nonzero_kept(self):
        """X + 0.001 should be kept - tests non-zero threshold."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        small_val = 0.001 * np.ones((2, 3), dtype=np.float32)
        initializers = [create_initializer("val", small_val)]

        add_node = helper.make_node("Add", inputs=["X", "val"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        # Add with non-zero should be kept
        add_nodes = [n for n in optimized.graph.node if n.op_type == "Add"]
        assert len(add_nodes) == 1

    def test_graph_optimization_multi_output(self):
        """Multiple outputs should be handled correctly - tests multi-output logic."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        W1 = np.ones((3, 2), dtype=np.float32)
        W2 = np.ones((3, 2), dtype=np.float32)
        initializers = [
            create_initializer("W1", W1),
            create_initializer("W2", W2),
        ]

        matmul1 = helper.make_node("MatMul", inputs=["X", "W1"], outputs=["Y1"])
        matmul2 = helper.make_node("MatMul", inputs=["X", "W2"], outputs=["Y2"])

        outputs = [
            create_tensor_value_info("Y1", "float32", [1, 2]),
            create_tensor_value_info("Y2", "float32", [1, 2]),
        ]
        model = create_minimal_onnx_model([matmul1, matmul2], inputs, outputs, initializers)

        optimized = optimize_onnx(model, has_batch_dim=True)
        assert optimized

    def test_numerical_correctness_after_chained_ops(self):
        """Graph outputs should be unchanged after chained operation optimization."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        inputs = [X]

        # Create model with chained ops: MatMul -> Add
        W = np.eye(3, 2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        initializers = [
            create_initializer("W", W),
            create_initializer("b", b),
        ]

        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_out"])
        add_node = helper.make_node("Add", inputs=["matmul_out", "b"], outputs=["Y"])

        outputs = [create_tensor_value_info("Y", "float32", [1, 2])]
        model = create_minimal_onnx_model([matmul_node, add_node], inputs, outputs, initializers)

        optimized = optimize_onnx(model, fuse_matmul_add=True, has_batch_dim=True)

        # Run both and compare
        test_input = np.ones((1, 3), dtype=np.float32)

        original_sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        original_out = np.asarray(original_sess.run(None, {"X": test_input})[0])

        optimized_sess = ort.InferenceSession(
            optimized.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        optimized_out = np.asarray(optimized_sess.run(None, {"X": test_input})[0])

        np.testing.assert_allclose(original_out, optimized_out, rtol=1e-5, atol=1e-6)
