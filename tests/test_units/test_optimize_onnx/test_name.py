"""Unit tests for slimonnx.optimize_onnx._name name-simplification helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._name import (
    _rename_initializers,
    _rename_io_nodes,
    _simplify_names,
    _update_node_input_names,
    _update_node_output_names,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import create_initializer, create_tensor_value_info


class TestRenameIoNodes:
    """_rename_io_nodes assigns input_<n>/output_<n> and returns mappings."""

    def test_inputs_and_outputs_get_sequential_names(self):
        inputs = [create_tensor_value_info("orig_in", "float32", [1])]
        outputs = [create_tensor_value_info("orig_out", "float32", [1])]

        in_map, out_map, counter = _rename_io_nodes(inputs, outputs, start_counter=0)

        assert inputs[0].name == "input_0"
        assert outputs[0].name == "output_1"
        assert in_map == {"orig_in": "input_0"}
        assert out_map == {"orig_out": "output_1"}
        assert counter == 2

    def test_counter_advances_from_start(self):
        inputs = [create_tensor_value_info("a", "float32", [1])]
        outputs = []
        _, _, counter = _rename_io_nodes(inputs, outputs, start_counter=10)

        assert inputs[0].name == "input_10"
        assert counter == 11


class TestUpdateNodeOutputNames:
    """_update_node_output_names renames node + outputs and updates mapping."""

    def test_single_output_uses_node_name(self):
        n = helper.make_node("Relu", inputs=["x"], outputs=["y"], name="orig")
        node_output_map: dict[str, str] = {}

        counter = _update_node_output_names(
            [n],
            output_old_new_mapping={},
            node_output_names_mapping=node_output_map,
            start_counter=5,
        )

        assert n.name == "Relu_5"
        assert list(n.output) == ["Relu_5"]
        assert node_output_map == {"y": "Relu_5"}
        assert counter == 6

    def test_multi_output_uses_indexed_suffix(self):
        n = helper.make_node("Split", inputs=["x"], outputs=["a", "b"], name="orig")
        node_output_map: dict[str, str] = {}

        _update_node_output_names(
            [n],
            output_old_new_mapping={},
            node_output_names_mapping=node_output_map,
            start_counter=0,
        )

        assert list(n.output) == ["Split_0_0", "Split_0_1"]
        assert node_output_map == {"a": "Split_0_0", "b": "Split_0_1"}

    def test_graph_output_keeps_renamed_external_name(self):
        """When a node feeds a graph output, the renamed graph-output name wins."""
        n = helper.make_node("Relu", inputs=["x"], outputs=["original_Y"], name="r")
        out_map = {"original_Y": "output_0"}
        node_output_map: dict[str, str] = {}

        _update_node_output_names([n], out_map, node_output_map, start_counter=1)

        assert list(n.output) == ["output_0"]
        assert node_output_map["original_Y"] == "output_0"


class TestUpdateNodeInputNames:
    """_update_node_input_names rewrites node inputs from a mapping."""

    @pytest.mark.parametrize(
        ("inputs", "mapping", "expected"),
        [
            pytest.param(
                ["old_a", "old_b"],
                {"old_a": "input_0", "old_b": "Relu_2"},
                ["input_0", "Relu_2"],
                id="known_inputs_remapped",
            ),
            pytest.param(
                ["old_a", "stray"],
                {"old_a": "input_0"},
                ["input_0", "stray"],
                id="unknown_input_left_unchanged",
            ),
        ],
    )
    def test_node_input_remapping(self, inputs, mapping, expected):
        """Test _update_node_input_names rewriting node inputs from a mapping."""
        n = helper.make_node("Add", inputs=inputs, outputs=["y"], name="n")
        _update_node_input_names([n], mapping)

        assert list(n.input) == expected


class TestRenameInitializers:
    """_rename_initializers maps initializers to Initializer_<n> and rewires."""

    def test_initializers_renamed_and_node_inputs_updated(self):
        init_w = create_initializer("W", np.ones(1, dtype=np.float32))
        init_b = create_initializer("B", np.ones(1, dtype=np.float32))
        node = helper.make_node("Add", inputs=["x", "W", "B"], outputs=["y"], name="n")

        new_inits = _rename_initializers([node], {"W": init_w, "B": init_b})

        assert list(new_inits.keys()) == ["Initializer_0", "Initializer_1"]
        assert init_w.name == "Initializer_0"
        assert init_b.name == "Initializer_1"
        assert list(node.input) == ["x", "Initializer_0", "Initializer_1"]


class TestSimplifyNames:
    """_simplify_names end-to-end: rename inputs, outputs, nodes, initializers."""

    def test_full_pipeline_produces_canonical_names(self):
        in_node = create_tensor_value_info("orig_X", "float32", [1, 1])
        out_node = create_tensor_value_info("orig_Y", "float32", [1, 1])
        weight = create_initializer("orig_W", np.ones((1, 1), dtype=np.float32))
        node = helper.make_node(
            "Gemm", inputs=["orig_X", "orig_W"], outputs=["orig_Y"], name="orig_gemm"
        )

        new_nodes, new_inits = _simplify_names([in_node], [out_node], [node], {"orig_W": weight})

        assert in_node.name == "input_0"
        assert out_node.name == "output_1"
        assert new_nodes[0].name == "Gemm_2"
        assert list(new_nodes[0].input) == ["input_0", "Initializer_0"]
        assert list(new_nodes[0].output) == ["output_1"]
        assert "Initializer_0" in new_inits
