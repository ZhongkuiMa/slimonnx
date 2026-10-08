"""Tests for redundant operation removal optimization."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import TensorProto, helper

from slimonnx.optimize_onnx._redundant import (
    _collapse_consecutive_reshapes,
    _is_redundant_arithmetic_op,
    _is_redundant_pad,
    _is_redundant_reshape_or_flatten,
    _remove_redundant_operations,
    _rewire_redundant_node,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestSkipRedundantNode:
    """Test _rewire_redundant_node function."""

    def test_rewire_redundant_node_rewires_input(self):
        """Test that skip_redundant_node rewires node connections."""
        # Create two nodes: Identity -> Relu
        identity_node = helper.make_node("Identity", inputs=["X"], outputs=["temp"])
        relu_node = helper.make_node("Relu", inputs=["temp"], outputs=["Y"])

        nodes_list = [identity_node, relu_node]
        output_info = [create_tensor_value_info("Z", "float32", [1, 3])]

        # Skip the identity node
        _rewire_redundant_node(identity_node, nodes_list, output_info)

        # Relu should now take X directly
        assert relu_node.input[0] == "X"

    def test_rewire_redundant_node_updates_graph_output(self):
        """Test that skip_redundant_node updates graph outputs."""
        identity_node = helper.make_node("Identity", inputs=["X"], outputs=["temp"])
        nodes_list = [identity_node]

        # Output info uses "temp" as name
        output_info = [create_tensor_value_info("Y", "float32", [1, 3])]
        output_info[0].name = "temp"

        # Skip the identity node
        _rewire_redundant_node(identity_node, nodes_list, output_info)

        # Output should be updated to X
        assert output_info[0].name == "X"


class TestCollapseConsecutiveReshapes:
    """Test _collapse_consecutive_reshapes function."""

    def test_collapse_single_reshape(self):
        """Test collapsing with single reshape (no collapse)."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 3])

        shape_init = create_initializer("shape", np.array([1, 3], dtype=np.int64))

        reshape_node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])

        model = create_minimal_onnx_model([reshape_node], [X], [Y], [shape_init])
        nodes = list(model.graph.node)

        initializers = {init.name: init for init in model.graph.initializer}
        result = _collapse_consecutive_reshapes(nodes, initializers, {"Y"})
        assert len(result) == 1
        assert result[0].op_type == "Reshape"

    def test_collapse_consecutive_reshapes(self):
        """Test collapsing three consecutive reshapes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        W = create_tensor_value_info("W", "float32", [6])

        shape1 = create_initializer("shape1", np.array([3, 2], dtype=np.int64))
        shape2 = create_initializer("shape2", np.array([6], dtype=np.int64))
        shape3 = create_initializer("shape3", np.array([6], dtype=np.int64))

        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["temp1"])
        reshape2 = helper.make_node("Reshape", inputs=["temp1", "shape2"], outputs=["temp2"])
        reshape3 = helper.make_node("Reshape", inputs=["temp2", "shape3"], outputs=["W"])

        model = create_minimal_onnx_model(
            [reshape1, reshape2, reshape3], [X], [W], [shape1, shape2, shape3]
        )
        nodes = list(model.graph.node)

        initializers = {init.name: init for init in model.graph.initializer}
        result = _collapse_consecutive_reshapes(nodes, initializers, {"W"})

        assert len(result) == 1
        assert result[0].input[0] == "X"

    def test_collapse_invalid_reshape_raises(self):
        """Middle of 3 Reshapes with 1 input triggers structural ValueError."""
        # Validation only fires when both pre_pre_node and pre_node are set,
        # so we need three consecutive Reshape nodes; the middle one is
        # malformed (1 input instead of 2).
        reshape1 = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["t1"])
        reshape2 = helper.make_node("Reshape", inputs=["t1"], outputs=["t2"])
        reshape3 = helper.make_node("Reshape", inputs=["t2", "shape3"], outputs=["Y"])

        with pytest.raises(ValueError, match="Invalid Reshape node structure"):
            _collapse_consecutive_reshapes([reshape1, reshape2, reshape3], {}, set())

    def test_does_not_collapse_disconnected_adjacent_reshapes(self):
        """Adjacency without a data edge does not authorize node removal."""
        shape1 = create_initializer("shape1", [3, 2], dtype="int64")
        shape2 = create_initializer("shape2", [6], dtype="int64")
        source = helper.make_node("Identity", inputs=["X"], outputs=["source"])
        first = helper.make_node("Reshape", inputs=["source", "shape1"], outputs=["middle"])
        second = helper.make_node("Reshape", inputs=["X", "shape2"], outputs=["Y"])

        result = _collapse_consecutive_reshapes(
            [source, first, second],
            {"shape1": shape1, "shape2": shape2},
            {"Y"},
        )

        assert result == [source, first, second]

    def test_does_not_collapse_branched_intermediate(self):
        """A reshape shared by two consumers must remain a graph producer."""
        shape1 = create_initializer("shape1", [3, 2], dtype="int64")
        shape2 = create_initializer("shape2", [6], dtype="int64")
        first = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["middle"])
        second = helper.make_node("Reshape", inputs=["middle", "shape2"], outputs=["Y"])
        branch = helper.make_node("Identity", inputs=["middle"], outputs=["Z"])

        result = _collapse_consecutive_reshapes(
            [first, second, branch],
            {"shape1": shape1, "shape2": shape2},
            {"Y", "Z"},
        )

        assert result == [first, second, branch]

    def test_does_not_collapse_public_intermediate(self):
        """An externally visible intermediate keeps its named producer."""
        shape1 = create_initializer("shape1", [3, 2], dtype="int64")
        shape2 = create_initializer("shape2", [6], dtype="int64")
        first = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["middle"])
        second = helper.make_node("Reshape", inputs=["middle", "shape2"], outputs=["Y"])

        result = _collapse_consecutive_reshapes(
            [first, second],
            {"shape1": shape1, "shape2": shape2},
            {"middle", "Y"},
        )

        assert result == [first, second]

    def test_does_not_collapse_input_relative_zero_target(self):
        """Default allowzero=0 copies dimensions from the immediate input."""
        shape1 = create_initializer("shape1", [3, 2], dtype="int64")
        shape2 = create_initializer("shape2", [0, 2], dtype="int64")
        first = helper.make_node("Reshape", inputs=["X", "shape1"], outputs=["middle"])
        second = helper.make_node("Reshape", inputs=["middle", "shape2"], outputs=["Y"])

        result = _collapse_consecutive_reshapes(
            [first, second],
            {"shape1": shape1, "shape2": shape2},
            {"Y"},
        )

        assert result == [first, second]

    def test_does_not_interpret_custom_domain_reshape(self):
        """A custom operator named Reshape has no ONNX reshape contract."""
        first = helper.make_node(
            "Reshape",
            inputs=["X"],
            outputs=["middle"],
            domain="example.custom",
        )
        second = helper.make_node(
            "Reshape",
            inputs=["middle"],
            outputs=["Y"],
            domain="example.custom",
        )

        result = _collapse_consecutive_reshapes([first, second], {}, {"Y"})

        assert result == [first, second]


class TestIsRedundantReshapeOrFlatten:
    """Test _is_redundant_reshape_or_flatten  # type: ignore function."""

    @pytest.mark.parametrize(
        ("op_type", "input_shape", "output_shape", "expected_redundant"),
        [
            ("Reshape", [1, 3, 4], [1, 3, 4], True),
            ("Reshape", [1, 3, 4], [1, 12], False),
            ("Flatten", [1, 3], [1, 3], True),
        ],
    )
    def test_shape_change_redundancy(self, op_type, input_shape, output_shape, expected_redundant):
        """Test redundancy detection for reshape and flatten operations."""
        data_shapes = {"input": input_shape, "output": output_shape}
        node = helper.make_node(op_type, inputs=["input"], outputs=["output"])
        is_redundant = _is_redundant_reshape_or_flatten(node, data_shapes)
        assert is_redundant == expected_redundant


class TestIsRedundantArithmeticOp:
    """Test _is_redundant_arithmetic_op function."""

    @pytest.mark.parametrize(
        (
            "op_type",
            "initializer_value",
            "init_name",
            "initializer_first",
            "expected_redundant",
        ),
        [
            ("Add", np.zeros(3, dtype=np.float32), "zero", False, True),
            ("Sub", np.zeros(3, dtype=np.float32), "zero", False, True),
            ("Mul", np.ones(3, dtype=np.float32), "ones", False, True),
            ("Div", np.ones(3, dtype=np.float32), "ones", False, True),
            ("Add", np.zeros(3, dtype=np.float32), "zero", True, True),
            ("Sub", np.zeros(3, dtype=np.float32), "zero", True, False),
            ("Mul", np.ones(3, dtype=np.float32), "ones", True, True),
            ("Div", np.ones(3, dtype=np.float32), "ones", True, False),
            (
                "Add",
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "const",
                False,
                False,
            ),
        ],
    )
    def test_operations_redundancy(
        self,
        op_type,
        initializer_value,
        init_name,
        initializer_first,
        expected_redundant,
    ):
        """Test redundancy detection for various arithmetic operations."""
        initializers = {init_name: create_initializer(init_name, initializer_value)}
        inputs = [init_name, "X"] if initializer_first else ["X", init_name]
        node = helper.make_node(op_type, inputs=inputs, outputs=["Y"])
        is_redundant, found_init = _is_redundant_arithmetic_op(node, initializers)
        assert is_redundant == expected_redundant
        if expected_redundant:
            assert found_init == init_name
        else:
            assert found_init is None

    def test_operation_without_initializers_not_redundant(self):
        """Test that op without initializers returns not redundant."""
        initializers: dict[str, Any] = {}
        add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
        is_redundant, init_name = _is_redundant_arithmetic_op(add_node, initializers)
        assert not is_redundant
        assert init_name is None


class TestIsRedundantPad:
    """Test _is_redundant_pad function."""

    def test_pad_with_all_zeros_is_redundant(self):
        """Test that Pad with all zeros is redundant."""
        pads = np.zeros(4, dtype=np.int64)
        initializers = {"pads": create_initializer("pads", pads)}

        pad_node = helper.make_node("Pad", inputs=["X", "pads"], outputs=["Y"])

        is_redundant = _is_redundant_pad(pad_node, initializers)
        assert is_redundant

    def test_pad_with_nonzero_not_redundant(self):
        """Test that Pad with non-zero values is not redundant."""
        pads = np.array([1, 0, 1, 0], dtype=np.int64)
        initializers = {"pads": create_initializer("pads", pads)}

        pad_node = helper.make_node("Pad", inputs=["X", "pads"], outputs=["Y"])

        is_redundant = _is_redundant_pad(pad_node, initializers)
        assert not is_redundant


class TestRemoveRedundantOperations:
    """Test _remove_redundant_operations  # type: ignore function."""

    @pytest.mark.parametrize(
        ("op_type", "initializer_name", "initializer_value"),
        [
            ("Add", "zero", np.zeros(3, dtype=np.float32)),
            ("Reshape", "shape", np.array([1, 3], dtype=np.int64)),
            ("Pad", "pads", np.zeros(4, dtype=np.int64)),
        ],
    )
    def test_removes_redundant_operation(self, op_type, initializer_name, initializer_value):
        """Test removing redundant operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Z = create_tensor_value_info("Z", "float32", [1, 3])

        initializers_list = [create_initializer(initializer_name, initializer_value)]
        node = helper.make_node(op_type, inputs=["X", initializer_name], outputs=["Y"])
        consumer = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([node, consumer], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3], "Y": [1, 3], "Z": [1, 3]}
        output_nodes = list(model.graph.output)

        result = _remove_redundant_operations(nodes, initializers_dict, data_shapes, output_nodes)  # type: ignore[arg-type]  # dict invariance
        assert len(result) == 1
        assert result[0].op_type == "Relu"
        assert result[0].input[0] == "X"
        assert initializer_name not in initializers_dict

    def test_keeps_redundant_node_that_defines_public_output(self):
        """Removing a public producer must not rename the graph output."""
        zero = create_initializer("zero", np.zeros(3, dtype=np.float32))
        add = helper.make_node("Add", inputs=["X", "zero"], outputs=["Y"])
        output = create_tensor_value_info("Y", "float32", [1, 3])
        initializers = {zero.name: zero}

        result = _remove_redundant_operations(
            [add],
            initializers,
            {"X": [1, 3], "Y": [1, 3]},
            [output],
        )

        assert result == [add]
        assert output.name == "Y"
        assert "zero" in initializers

    def test_preserves_identity_initializer_used_by_another_node(self):
        """An initializer remains while any retained node still consumes it."""
        zero = create_initializer("zero", np.zeros(3, dtype=np.float32))
        add = helper.make_node("Add", inputs=["X", "zero"], outputs=["temp"])
        subtract = helper.make_node("Sub", inputs=["zero", "X"], outputs=["Y"])
        output = create_tensor_value_info("Y", "float32", [1, 3])
        initializers = {zero.name: zero}

        result = _remove_redundant_operations(
            [add, subtract],
            initializers,
            {"X": [1, 3], "temp": [1, 3], "Y": [1, 3]},
            [output],
        )

        assert result == [subtract]
        assert "zero" in initializers

    def test_keeps_custom_domain_arithmetic(self):
        """An identically named custom operator has no ONNX identity contract."""
        zero = create_initializer("zero", np.zeros(3, dtype=np.float32))
        custom = helper.make_node(
            "Add",
            inputs=["X", "zero"],
            outputs=["temp"],
            domain="example.custom",
        )
        consumer = helper.make_node("Relu", inputs=["temp"], outputs=["Y"])
        initializers = {zero.name: zero}

        result = _remove_redundant_operations(
            [custom, consumer],
            initializers,
            {"X": [1, 3], "temp": [1, 3], "Y": [1, 3]},
            [create_tensor_value_info("Y", "float32", [1, 3])],
        )

        assert result == [custom, consumer]
        assert "zero" in initializers

    def test_keep_non_redundant_operations(self):
        """Test that non-redundant operations are kept."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Y = create_tensor_value_info("Y", "float32", [1, 4])

        # Non-zero add
        non_zero = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        initializers_list = [create_initializer("const", non_zero)]

        add_node = helper.make_node("Add", inputs=["X", "const"], outputs=["Y"])

        model = create_minimal_onnx_model([add_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3], "Y": [1, 3]}
        output_nodes = list(model.graph.output)

        result = _remove_redundant_operations(nodes, initializers_dict, data_shapes, output_nodes)  # type: ignore[arg-type]  # dict invariance

        # Should keep the Add node
        assert len(result) == 1
        assert result[0].op_type == "Add"

    @pytest.mark.parametrize(
        ("source_type", "target_type", "removed"),
        [
            (TensorProto.FLOAT, TensorProto.FLOAT, True),
            (TensorProto.FLOAT, TensorProto.DOUBLE, False),
        ],
    )
    def test_remove_only_element_type_preserving_cast(
        self,
        source_type: int,
        target_type: int,
        removed: bool,
    ) -> None:
        """A Cast is an identity only when type inference proves equal types."""
        cast = helper.make_node("Cast", inputs=["X"], outputs=["cast_out"], to=target_type)
        relu = helper.make_node("Relu", inputs=["cast_out"], outputs=["Y"])
        output = create_tensor_value_info("Y", "float32", [1, 3])

        result = _remove_redundant_operations(
            [cast, relu],
            {},
            {"X": [1, 3], "cast_out": [1, 3], "Y": [1, 3]},
            [output],
            {"X": source_type, "cast_out": target_type},
        )

        assert (cast not in result) is removed
        assert relu.input[0] == ("X" if removed else "cast_out")

    def test_remove_multiple_redundant_operations(self):
        """Test removing multiple redundant operations."""
        X = create_tensor_value_info("X", "float32", [1, 3])
        Z = create_tensor_value_info("Z", "float32", [1, 3])

        zero = np.zeros(3, dtype=np.float32)
        initializers_list = [create_initializer("zero", zero)]

        add_node = helper.make_node("Add", inputs=["X", "zero"], outputs=["temp"])
        sub_node = helper.make_node("Sub", inputs=["temp", "zero"], outputs=["Z"])

        model = create_minimal_onnx_model([add_node, sub_node], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        data_shapes = {"X": [1, 3], "temp": [1, 3], "Z": [1, 3]}
        output_nodes = list(model.graph.output)

        result = _remove_redundant_operations(nodes, initializers_dict, data_shapes, output_nodes)  # type: ignore[arg-type]  # dict invariance

        # Should remove the Add node (redundant), but Sub node remains
        # because it still references temp (which is now rewired to X)
        assert len(result) <= len(nodes)
        # At least one node should be removed
        assert len(result) < len(nodes)
