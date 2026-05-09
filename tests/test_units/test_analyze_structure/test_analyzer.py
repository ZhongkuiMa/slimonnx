"""Tests for ONNX model structure analyzer."""

__docformat__ = "restructuredtext"

import pytest
from onnx import helper

from slimonnx.structure_analysis.analyzer import (
    analyze_inputs_outputs,
    analyze_structure,
    count_op_types,
)
from tests.test_units.conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestCountOpTypes:
    """Tests for count_op_types."""

    def test_returns_op_type_counts(self):
        """count_op_types returns correct counts for multiple op types."""
        nodes = [
            helper.make_node("Relu", inputs=["X"], outputs=["Y"]),
            helper.make_node("Relu", inputs=["Y"], outputs=["Z"]),
            helper.make_node("Add", inputs=["A", "B"], outputs=["C"]),
        ]
        result = count_op_types(nodes)
        assert result == {"Relu": 2, "Add": 1}

    def test_returns_empty_for_empty_list(self):
        """count_op_types returns empty dict for empty node list."""
        result = count_op_types([])
        assert result == {}

    def test_returns_single_type(self):
        """count_op_types returns correct count for single op type."""
        nodes = [helper.make_node("Relu", inputs=["X"], outputs=["Y"])]
        result = count_op_types(nodes)
        assert result == {"Relu": 1}


class TestAnalyzeInputsOutputs:
    """Tests for analyze_inputs_outputs."""

    def test_returns_input_output_metadata(self):
        """analyze_inputs_outputs returns input/output metadata for a valid model."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        model = create_minimal_onnx_model([], inputs, outputs)

        result = analyze_inputs_outputs(model)

        assert result["num_inputs"] == 1
        assert result["num_outputs"] == 1
        assert len(result["inputs"]) == 1
        assert len(result["outputs"]) == 1
        assert result["inputs"][0]["name"] == "X"
        assert result["outputs"][0]["name"] == "Y"

    def test_handles_multiple_inputs_outputs(self):
        """analyze_inputs_outputs handles models with multiple inputs and outputs."""
        inputs = [
            create_tensor_value_info("A", "float32", [2, 3]),
            create_tensor_value_info("B", "float32", [2, 3]),
        ]
        outputs = [
            create_tensor_value_info("C", "float32", [2, 3]),
            create_tensor_value_info("D", "float32", [2, 3]),
        ]
        model = create_minimal_onnx_model([], inputs, outputs)

        result = analyze_inputs_outputs(model)

        assert result["num_inputs"] == 2
        assert result["num_outputs"] == 2

    def test_returns_shape_info(self):
        """analyze_inputs_outputs returns shape for each input/output."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3, 4])]
        outputs = [create_tensor_value_info("Y", "float32", [5, 6])]
        model = create_minimal_onnx_model([], inputs, outputs)

        result = analyze_inputs_outputs(model)

        assert len(result["inputs"]) > 0
        assert result["inputs"][0]["shape"] == [2, 3, 4]
        assert len(result["outputs"]) > 0
        assert result["outputs"][0]["shape"] == [5, 6]


class TestAnalyzeStructure:
    """Tests for analyze_structure."""

    def test_returns_structure_metadata(self):
        """analyze_structure returns structure metadata for a valid model."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], inputs, outputs)

        result = analyze_structure(model)

        assert result["node_count"] == 1
        assert result["op_type_counts"] == {"Identity": 1}
        assert "initializer_count" in result
        assert result["num_inputs"] == 1
        assert result["num_outputs"] == 1

    @pytest.mark.parametrize(
        ("data_shapes", "expected_has_shapes"),
        [
            ({"X": [2, 3]}, True),
            pytest.param(None, False, id="data_shapes_omitted"),
        ],
    )
    def test_has_shapes_flag(self, data_shapes, expected_has_shapes):
        """analyze_structure correctly reports has_shapes flag."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], inputs, outputs)

        kwargs = {} if data_shapes is None else {"data_shapes": data_shapes}
        result = analyze_structure(model, **kwargs)

        assert result["has_shapes"] is expected_has_shapes
        # [REVIEW] Merged into test_has_shapes_flag via parametrize — original: test_accepts_data_shapes, test_reports_no_shapes_when_omitted
