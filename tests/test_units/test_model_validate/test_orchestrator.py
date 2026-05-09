"""Tests for model validation orchestration."""

__docformat__ = "restructuredtext"

import pytest
from onnx import helper

from slimonnx.model_validate._orchestrator import validate_model
from tests.test_units.conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestValidateModel:
    """Tests for validate_model."""

    def test_returns_dict_with_expected_keys(self):
        """validate_model returns a dict with expected validation keys."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], inputs, outputs)

        result = validate_model(model)

        assert isinstance(result, dict)
        assert "onnx_checker" in result
        assert "runtime" in result
        assert "dead_nodes" in result
        assert "broken_connections" in result
        assert "orphan_initializers" in result
        assert "type_errors" in result
        assert "shape_errors" in result
        assert "is_valid" in result

    def test_raises_for_broken_model(self):
        """validate_model raises ValidationError for a model with disconnected nodes."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        node = helper.make_node("Add", inputs=["A", "B"], outputs=["Y"])
        model = create_minimal_onnx_model([node], inputs, outputs)

        with pytest.raises(Exception, match=r"must be topologically sorted|not output of any"):
            validate_model(model)

    def test_accepts_data_shapes(self):
        """validate_model accepts optional data_shapes parameter."""
        inputs = [create_tensor_value_info("X", "float32", [2, 3])]
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]
        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], inputs, outputs)

        result = validate_model(model, data_shapes={"X": [2, 3]})

        assert isinstance(result, dict)
        assert "shape_errors" in result
