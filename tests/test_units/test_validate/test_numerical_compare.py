"""Tests for numerical comparison utilities."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_minimal_onnx_model,
    create_tensor_value_info,
)

from slimonnx.model_validate.numerical_compare import (
    _compare_outputs,
    _generate_test_inputs,
    _load_test_inputs_from_file,
    _map_inputs_for_model2,
    generate_inputs_from_bounds,
    run_onnx_inference,
)


class TestGenerateInputsFromBounds:
    """Test generate_inputs_from_bounds function."""

    def test_generate_inputs_from_bounds_single_sample(self):
        """Test generating single input from bounds."""
        bounds = ([0.0], [1.0])
        shape = (2, 3)
        result = generate_inputs_from_bounds(bounds, shape, num_samples=1)
        assert len(result) == 1
        assert "input" in result[0]
        assert result[0]["input"].shape == (2, 3)
        assert result[0]["input"].dtype == np.float32

    def test_generate_inputs_from_bounds_multiple_samples(self):
        """Test generating multiple inputs from bounds."""
        bounds = ([0.0, 0.0], [1.0, 1.0])
        shape = (2,)
        result = generate_inputs_from_bounds(bounds, shape, num_samples=5)
        assert len(result) == 5
        for inputs in result:
            assert inputs["input"].shape == (2,)
            assert np.all(inputs["input"] >= 0.0)
            assert np.all(inputs["input"] <= 1.0)

    def test_generate_inputs_from_bounds_values_in_range(self):
        """Test that generated inputs are within bounds."""
        bounds = ([-10.0, -5.0, 0.0], [10.0, 5.0, 1.0])
        shape = (3,)
        result = generate_inputs_from_bounds(bounds, shape, num_samples=10)
        for inputs in result:
            values = inputs["input"]
            assert values.shape == (3,)
            assert np.all(values >= np.array([-10.0, -5.0, 0.0]))
            assert np.all(values <= np.array([10.0, 5.0, 1.0]))

    def test_generate_inputs_from_bounds_mismatched_bounds(self):
        """Test error when bounds have different lengths."""
        bounds = ([0.0], [1.0, 2.0])
        shape = (2,)
        with pytest.raises(ValueError, match="same length"):
            generate_inputs_from_bounds(bounds, shape)


class TestMapInputsForModel2:
    """Test _map_inputs_for_model2 function."""

    def test_map_inputs_for_model2_same_names(self):
        """Test mapping when input names are same."""
        inputs = {"input1": np.array([1.0, 2.0]), "input2": np.array([3.0, 4.0])}
        input_names1 = ["input1", "input2"]
        input_names2 = ["input1", "input2"]
        result = _map_inputs_for_model2(inputs, input_names1, input_names2)
        assert result == inputs

    def test_map_inputs_for_model2_different_names(self):
        """Test mapping when input names differ."""
        inputs = {"X": np.array([1.0, 2.0])}
        input_names1 = ["X"]
        input_names2 = ["data"]
        result = _map_inputs_for_model2(inputs, input_names1, input_names2)
        assert "data" in result
        assert np.array_equal(result["data"], inputs["X"])
        assert "X" not in result

    def test_map_inputs_for_model2_generic_input_name(self):
        """Test mapping with generic 'input' name."""
        inputs = {"input": np.array([1.0, 2.0])}
        input_names1 = ["input"]
        input_names2 = ["data"]
        result = _map_inputs_for_model2(inputs, input_names1, input_names2)
        assert "data" in result
        assert np.array_equal(result["data"], inputs["input"])

    def test_map_inputs_for_model2_multiple_inputs(self):
        """Test mapping multiple inputs."""
        inputs = {"in1": np.array([1.0]), "in2": np.array([2.0])}
        input_names1 = ["in1", "in2"]
        input_names2 = ["x", "y"]
        result = _map_inputs_for_model2(inputs, input_names1, input_names2)
        assert "x" in result
        assert "y" in result
        assert np.array_equal(result["x"], inputs["in1"])
        assert np.array_equal(result["y"], inputs["in2"])


class TestCompareOutputs:
    """Test _compare_outputs function."""

    def test_compare_outputs_matching(self):
        """Test comparing matching outputs."""
        outputs1 = {"output": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"output": np.array([1.0, 2.0, 3.0])}
        output_names1 = ["output"]
        output_names2 = ["output"]
        match, _, mismatches = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is True
        assert len(mismatches) == 0

    def test_compare_outputs_within_tolerance(self):
        """Test outputs within tolerance are considered matching."""
        outputs1 = {"output": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"output": np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])}
        output_names1 = ["output"]
        output_names2 = ["output"]
        match, _, _ = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is True

    def test_compare_outputs_exceeds_tolerance(self):
        """Test outputs exceeding tolerance are not matching."""
        outputs1 = {"output": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"output": np.array([1.0 + 1e-3, 2.0, 3.0])}
        output_names1 = ["output"]
        output_names2 = ["output"]
        match, max_diff, mismatches = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is False
        assert max_diff > 0.0
        assert len(mismatches) > 0

    def test_compare_outputs_missing_output(self):
        """Test error when output is missing from one model."""
        outputs1 = {"output1": np.array([1.0, 2.0])}
        outputs2: dict[str, Any] = {}  # Missing output2
        output_names1 = ["output1"]
        output_names2 = ["output2"]
        match, _, mismatches = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is False
        assert len(mismatches) > 0

    def test_compare_outputs_multiple_outputs(self):
        """Test comparing multiple outputs."""
        outputs1 = {"out1": np.array([1.0]), "out2": np.array([2.0])}
        outputs2 = {"out1": np.array([1.0]), "out2": np.array([2.0])}
        output_names1 = ["out1", "out2"]
        output_names2 = ["out1", "out2"]
        match, _, mismatches = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is True
        assert len(mismatches) == 0

    def test_compare_outputs_one_output_differs(self):
        """Test when one output differs among multiple."""
        outputs1 = {"out1": np.array([1.0]), "out2": np.array([2.0])}
        outputs2 = {"out1": np.array([1.0]), "out2": np.array([2.5])}
        output_names1 = ["out1", "out2"]
        output_names2 = ["out1", "out2"]
        match, max_diff, mismatches = _compare_outputs(
            0, outputs1, outputs2, output_names1, output_names2, rtol=1e-5, atol=1e-6
        )
        assert match is False
        assert max_diff > 0.0
        assert len(mismatches) == 1


class TestGenerateTestInputs:
    """Test _generate_test_inputs function."""

    def test_generate_test_inputs_with_bounds(self, tmp_path):
        """Test generating inputs with bounds."""
        # Create a simple model with 1D input
        X = create_tensor_value_info("X", "float32", [3])
        Y = create_tensor_value_info("Y", "float32", [3])
        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([relu], [X], [Y])
        model_path = tmp_path / "model.onnx"
        from onnx import save

        save(model, str(model_path))

        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        result = _generate_test_inputs(str(model_path), bounds, num_samples=2)
        assert len(result) == 2
        assert all("input" in r for r in result)

    def test_generate_test_inputs_without_bounds(self, tmp_path):
        """Test generating random inputs without bounds."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])
        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([relu], [X], [Y])
        model_path = tmp_path / "model.onnx"
        from onnx import save

        save(model, str(model_path))

        result = _generate_test_inputs(str(model_path), None, num_samples=2)
        assert len(result) == 2
        assert all("input" in r or len(r) > 0 for r in result)


class TestLoadTestInputsFromFile:
    """Test _load_test_inputs_from_file function."""

    def test_load_test_inputs_from_npy_1d(self, tmp_path):
        """Test loading 1D array from .npy file."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, data)

        result = _load_test_inputs_from_file(str(npy_path), num_samples=5)
        assert result is not None
        assert len(result) == 1
        assert "input" in result[0]
        assert np.array_equal(result[0]["input"], data)

    def test_load_test_inputs_from_npy_2d(self, tmp_path):
        """Test loading 2D array from .npy file."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, data)

        result = _load_test_inputs_from_file(str(npy_path), num_samples=2)
        assert result is not None
        assert len(result) == 2
        assert np.array_equal(result[0]["input"], data[0])
        assert np.array_equal(result[1]["input"], data[1])

    def test_load_test_inputs_from_npz_with_inputs(self, tmp_path):
        """Test loading from .npz file with 'inputs' key."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, inputs=data)

        result = _load_test_inputs_from_file(str(npz_path), num_samples=2)
        assert result is not None
        assert len(result) == 2

    def test_load_test_inputs_from_npz_with_x(self, tmp_path):
        """Test loading from .npz file with 'X' key."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, X=data)

        result = _load_test_inputs_from_file(str(npz_path), num_samples=2)
        assert result is not None
        assert len(result) == 2

    def test_load_test_inputs_from_npz_no_inputs(self, tmp_path):
        """Test loading from .npz file without expected keys."""
        npz_path = tmp_path / "test.npz"
        np.savez(npz_path, unknown=np.array([1.0, 2.0]))

        result = _load_test_inputs_from_file(str(npz_path), num_samples=5)
        assert result is None

    def test_load_test_inputs_unsupported_format(self, tmp_path):
        """Test error with unsupported file format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("test")

        result = _load_test_inputs_from_file(str(txt_path), num_samples=5)
        assert result is None

    def test_load_test_inputs_nonexistent_file(self):
        """Test error with non-existent file."""
        result = _load_test_inputs_from_file("/nonexistent/path/test.npy", num_samples=5)
        assert result is None

    def test_load_test_inputs_num_samples_limit(self, tmp_path):
        """Test limiting number of loaded samples."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(npy_path, data)

        result = _load_test_inputs_from_file(str(npy_path), num_samples=2)
        assert result is not None
        assert len(result) == 2


class TestRunOnnxInference:
    """Test run_onnx_inference function."""

    def test_run_onnx_inference_simple_model(self, tmp_path):
        """Test running inference on simple model."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])
        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([relu], [X], [Y])
        model_path = tmp_path / "model.onnx"
        from onnx import save

        save(model, str(model_path))

        inputs = {"input": np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)}
        result = run_onnx_inference(str(model_path), inputs)
        assert "Y" in result
        assert result["Y"].shape == (2, 3)

    def test_run_onnx_inference_with_explicit_input_name(self, tmp_path):
        """Test inference with explicit input name."""
        X = create_tensor_value_info("data", "float32", [2, 3])
        Y = create_tensor_value_info("output", "float32", [2, 3])
        relu = helper.make_node("Relu", inputs=["data"], outputs=["output"])
        model = create_minimal_onnx_model([relu], [X], [Y])
        model_path = tmp_path / "model.onnx"
        from onnx import save

        save(model, str(model_path))

        inputs = {"data": np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)}
        result = run_onnx_inference(str(model_path), inputs)
        assert "output" in result
