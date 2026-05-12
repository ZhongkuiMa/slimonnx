"""Unit tests for slimonnx.model_validate.graph_validator."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.model_validate.graph_validator import (
    check_broken_connections,
    check_dead_nodes,
    check_orphan_initializers,
    check_shape_consistency,
    check_type_consistency,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import create_initializer, create_tensor_value_info  # type: ignore[import-not-found]


class TestCheckDeadNodes:
    """Tests for check_dead_nodes — backward reachability from outputs."""

    @pytest.mark.parametrize(
        ("nodes", "expected"),
        [
            (
                [
                    helper.make_node("Relu", inputs=["X"], outputs=["a"], name="n1"),
                    helper.make_node("Relu", inputs=["a"], outputs=["Y"], name="n2"),
                ],
                [],
            ),
            (
                [
                    helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="live"),
                    helper.make_node("Relu", inputs=["X"], outputs=["unused"], name="dead"),
                ],
                ["dead"],
            ),
        ],
    )
    def test_dead_node_detection(self, nodes, expected):
        """Test dead node detection in various scenarios."""
        outputs = [create_tensor_value_info("Y", "float32", [1])]
        result = check_dead_nodes(nodes, outputs)
        assert result == expected
        # [REVIEW] Merged into test_dead_node_detection via parametrize — original: test_all_nodes_reachable, test_dead_node_reported

    def test_unnamed_dead_node_uses_op_fallback(self):
        """Dead node without a name is reported as ``<op>_unnamed``."""
        live = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="live")
        dead = helper.make_node("Sigmoid", inputs=["X"], outputs=["unused"])
        outputs = [create_tensor_value_info("Y", "float32", [1])]

        assert check_dead_nodes([live, dead], outputs) == ["Sigmoid_unnamed"]


class TestCheckBrokenConnections:
    """Tests for check_broken_connections — input availability check."""

    def test_clean_graph_has_no_broken_connections(self):
        n = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="r")
        inputs = [create_tensor_value_info("X", "float32", [1])]
        assert check_broken_connections([n], {}, inputs) == []

    def test_node_with_missing_input_reported(self):
        n = helper.make_node("Add", inputs=["X", "missing"], outputs=["Y"], name="bad")
        inputs = [create_tensor_value_info("X", "float32", [1])]

        result = check_broken_connections([n], {}, inputs)

        assert len(result) == 1
        assert result[0]["node"] == "bad"
        assert result[0]["op_type"] == "Add"
        assert result[0]["missing_input"] == "missing"

    def test_initializers_count_as_available(self):
        n = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="ok")
        inputs = [create_tensor_value_info("X", "float32", [1])]
        initializers = {"W": create_initializer("W", np.ones(1, dtype=np.float32))}

        assert check_broken_connections([n], initializers, inputs) == []

    def test_empty_input_string_skipped(self):
        """Optional empty-string inputs (ONNX convention) are not flagged."""
        n = helper.make_node("Conv", inputs=["X", "W", ""], outputs=["Y"], name="conv")
        inputs = [create_tensor_value_info("X", "float32", [1, 1, 1, 1])]
        initializers = {"W": create_initializer("W", np.ones((1, 1, 1, 1), dtype=np.float32))}

        assert check_broken_connections([n], initializers, inputs) == []


class TestCheckOrphanInitializers:
    """Tests for check_orphan_initializers."""

    def test_used_initializer_not_orphan(self):
        n = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="n")
        initializers = {"W": create_initializer("W", np.ones(1, dtype=np.float32))}
        assert check_orphan_initializers([n], initializers) == []

    def test_unused_initializer_reported(self):
        n = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="n")
        initializers = {
            "W_used": create_initializer("W_used", np.ones(1, dtype=np.float32)),
            "W_orphan": create_initializer("W_orphan", np.ones(1, dtype=np.float32)),
        }
        # neither is referenced by Relu, so both are orphans
        assert sorted(check_orphan_initializers([n], initializers)) == ["W_orphan", "W_used"]


class TestCheckTypeConsistency:
    """Tests for check_type_consistency — flags UNDEFINED tensor types."""

    def test_well_typed_initializers_have_no_errors(self):
        initializers = {
            "W": create_initializer("W", np.ones(1, dtype=np.float32)),
        }
        assert check_type_consistency([], initializers) == []

    def test_undefined_data_type_reported(self):
        init = create_initializer("bad", np.ones(1, dtype=np.float32))
        init.data_type = 0  # UNDEFINED

        result = check_type_consistency([], {"bad": init})

        assert result == [{"tensor": "bad", "error": "Undefined data type"}]


class TestCheckShapeConsistency:
    """Tests for check_shape_consistency — flags unknown input/output shapes."""

    def test_complete_shapes_have_no_errors(self):
        n = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="n")
        data_shapes = {"X": [1], "Y": [1]}
        assert check_shape_consistency([n], data_shapes) == []

    @pytest.mark.parametrize(
        ("data_shapes", "assert_key", "assert_val"),
        [
            pytest.param({"Y": [1]}, "input", "X", id="missing_input"),
            pytest.param({"X": [1]}, "output", "Y", id="missing_output"),
        ],
    )
    def test_missing_shape_reported(self, data_shapes, assert_key, assert_val):
        """Test that missing input or output shape is reported."""
        n = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="n")

        result = check_shape_consistency([n], data_shapes)

        assert any(
            err[assert_key] == assert_val and err["error"] == "Unknown shape" for err in result
        )
