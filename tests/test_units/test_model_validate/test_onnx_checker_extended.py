"""Extended tests for ONNX checker validation."""

import sys
from pathlib import Path
from unittest import mock

from onnx import helper

from slimonnx.model_validate.onnx_checker import run_onnx_checker

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import create_tensor_value_info


class TestOnnxCheckerExtended:
    """Test run_onnx_checker exception handling paths."""

    def test_checker_catches_value_error(self):
        """Test ValueError exception handling in onnx_checker."""
        # Create a minimal valid model
        X = create_tensor_value_info("X", "float32", [1, 3, 224, 224])
        Y = create_tensor_value_info("Y", "float32", [1, 1000])

        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        # Mock onnx.checker.check_model to raise ValueError
        with mock.patch("onnx.checker.check_model") as mock_check:
            mock_check.side_effect = ValueError("Invalid model structure")

            result = run_onnx_checker(model)

            assert result["valid"] is False
            assert "Invalid model structure" in result["error"]

    def test_checker_catches_attribute_error(self):
        """Test AttributeError exception handling in onnx_checker."""
        # Create a minimal valid model
        X = create_tensor_value_info("X", "float32", [1, 3, 224, 224])
        Y = create_tensor_value_info("Y", "float32", [1, 1000])

        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        # Mock onnx.checker.check_model to raise AttributeError
        with mock.patch("onnx.checker.check_model") as mock_check:
            mock_check.side_effect = AttributeError("Missing required attribute")

            result = run_onnx_checker(model)

            assert result["valid"] is False
            assert "Missing required attribute" in result["error"]

    def test_checker_catches_type_error(self):
        """Test TypeError exception handling in onnx_checker."""
        # Create a minimal valid model
        X = create_tensor_value_info("X", "float32", [1, 3, 224, 224])
        Y = create_tensor_value_info("Y", "float32", [1, 1000])

        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        # Mock onnx.checker.check_model to raise TypeError
        with mock.patch("onnx.checker.check_model") as mock_check:
            mock_check.side_effect = TypeError("Type mismatch in model")

            result = run_onnx_checker(model)

            assert result["valid"] is False
            assert "Type mismatch in model" in result["error"]

    def test_checker_success_path(self):
        """Test successful ONNX model validation."""
        # Create a valid model
        X = create_tensor_value_info("X", "float32", [1, 3, 224, 224])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 224, 224])

        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        # This should not raise any exception
        result = run_onnx_checker(model)

        assert result["valid"] is True
        assert result["error"] is None

    def test_checker_returns_correct_structure(self):
        """Test that checker returns correct dictionary structure."""
        # Create a valid model
        X = create_tensor_value_info("X", "float32", [1])
        Y = create_tensor_value_info("Y", "float32", [1])

        node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = helper.make_graph([node], "test_graph", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        result = run_onnx_checker(model)

        # Verify result structure
        assert isinstance(result, dict)
        assert "valid" in result
        assert "error" in result
        assert isinstance(result["valid"], bool)
