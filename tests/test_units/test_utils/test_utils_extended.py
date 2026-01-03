"""Extended tests for ONNX utility functions - missing coverage."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from slimonnx.utils import (
    compare_outputs,
    convert_constant_to_initializer,
    generate_random_inputs,
    load_test_data_from_file,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConvertConstantToInitializerExtended:
    """Extended tests for convert_constant_to_initializer."""

    def test_convert_constant_node_float(self):
        """Test converting a Constant node to initializer."""
        # Create a Constant node with proper attributes
        const_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = numpy_helper.from_array(const_value, "const_out")
        attr = helper.make_attribute("value", tensor_proto)
        const_node = helper.make_node("Constant", inputs=[], outputs=["const_out"])
        const_node.attribute.append(attr)

        nodes = [const_node]
        initializers: dict[str, TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)

        # Constant node should be removed from nodes list
        assert len(result) == 0
        # Initializer should be added
        assert "const_out" in initializers
        assert np.allclose(numpy_helper.to_array(initializers["const_out"]), const_value)

    def test_convert_mixed_nodes(self):
        """Test converting with mixed Constant and other nodes."""
        # Create nodes
        const_value = np.array([1.0, 2.0], dtype=np.float32)
        tensor_proto = numpy_helper.from_array(const_value, "const_out")
        attr = helper.make_attribute("value", tensor_proto)
        const_node = helper.make_node("Constant", inputs=[], outputs=["const_out"])
        const_node.attribute.append(attr)
        relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        nodes = [const_node, relu_node]
        initializers: dict[str, TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)

        # Only Relu should remain
        assert len(result) == 1
        assert result[0].op_type == "Relu"
        # Constant converted to initializer
        assert "const_out" in initializers


class TestGenerateRandomInputsExtended:
    """Extended tests for generate_random_inputs with different dtypes."""

    def test_generate_float16_input(self):
        """Test generating float16 input."""
        # Create model with float16 input
        graph = helper.make_graph(
            [helper.make_node("Relu", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.float16
        assert result[0]["X"].shape == (2, 3)

    def test_generate_float64_input(self):
        """Test generating float64 (double) input."""
        # Create model with float64 input
        graph = helper.make_graph(
            [helper.make_node("Relu", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.DOUBLE, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.float64
        assert result[0]["X"].shape == (2, 3)

    def test_generate_int32_input(self):
        """Test generating int32 input."""
        # Create model with int32 input
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.INT32, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.INT32, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.int32
        assert result[0]["X"].shape == (2, 3)

    def test_generate_int64_input(self):
        """Test generating int64 input."""
        # Create model with int64 input
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.INT64, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.INT64, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.int64
        assert result[0]["X"].shape == (2, 3)

    def test_generate_uint8_input(self):
        """Test generating uint8 input."""
        # Create model with uint8 input
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.UINT8, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.uint8
        assert result[0]["X"].shape == (2, 3)

    def test_generate_int8_input(self):
        """Test generating int8 input."""
        # Create model with int8 input
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.INT8, [2, 3])],
            [helper.make_tensor_value_info("Y", TensorProto.INT8, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np.int8
        assert result[0]["X"].shape == (2, 3)

    def test_generate_unknown_dtype_defaults_to_float32(self):
        """Test that unknown dtype defaults to float32."""
        # Create model with unknown dtype (999)
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", 999, [2, 3])],  # Unknown type
            [helper.make_tensor_value_info("Y", 999, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        # Should default to float32
        assert result[0]["X"].dtype == np.float32


class TestLoadTestDataFromFile:
    """Test load_test_data_from_file function."""

    def test_load_npy_1d_array(self):
        """Test loading 1D numpy array from .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            np.save(temp_path, data)

        try:
            result = load_test_data_from_file(temp_path)
            assert len(result) == 1
            assert "input" in result[0]
            assert np.allclose(result[0]["input"], data)
        finally:
            Path(temp_path).unlink()

    def test_load_npy_2d_array(self):
        """Test loading 2D numpy array from .npy file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_path = f.name
            data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(temp_path, data)

        try:
            result = load_test_data_from_file(temp_path)
            assert len(result) == 2  # One array per row
            assert "input" in result[0]
            assert np.allclose(result[0]["input"], data[0])
        finally:
            Path(temp_path).unlink()

    def test_load_npz_with_inputs_key(self):
        """Test loading .npz file with 'inputs' key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
            inputs_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.savez(temp_path, inputs=inputs_data)

        try:
            result = load_test_data_from_file(temp_path)
            assert len(result) == 2
            assert "input" in result[0]
            assert np.allclose(result[0]["input"], inputs_data[0])
        finally:
            Path(temp_path).unlink()

    def test_load_npz_with_x_key(self):
        """Test loading .npz file with 'X' key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
            x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.savez(temp_path, X=x_data)

        try:
            result = load_test_data_from_file(temp_path)
            assert len(result) == 2
            assert "input" in result[0]
            assert np.allclose(result[0]["input"], x_data[0])
        finally:
            Path(temp_path).unlink()

    def test_load_npz_with_arbitrary_key(self):
        """Test loading .npz file with arbitrary key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name
            custom_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.savez(temp_path, custom_key=custom_data)

        try:
            result = load_test_data_from_file(temp_path)
            assert len(result) == 2
            assert "input" in result[0]
            assert np.allclose(result[0]["input"], custom_data[0])
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_test_data_from_file("/nonexistent/path/data.npy")

    def test_load_unsupported_format_raises_error(self):
        """Test that unsupported file format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name
            f.write(b"test data")

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_test_data_from_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestCompareOutputsExtended:
    """Extended tests for compare_outputs function."""

    def test_compare_missing_output_in_outputs2(self):
        """Test comparing when output missing in outputs2."""
        outputs1 = {"Y1": np.array([1.0, 2.0])}
        outputs2: dict[str, np.ndarray] = {}

        match, mismatches = compare_outputs(outputs1, outputs2)

        assert match is False
        assert len(mismatches) > 0
        assert any(m.get("type") == "missing_key" for m in mismatches)
        assert any("missing in outputs2" in m.get("message", "") for m in mismatches)

    def test_compare_missing_output_in_outputs1(self):
        """Test comparing when output missing in outputs1."""
        outputs1: dict[str, np.ndarray] = {}
        outputs2 = {"Y1": np.array([1.0, 2.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)

        assert match is False
        assert len(mismatches) > 0
        assert any(m.get("type") == "missing_key" for m in mismatches)
        assert any("missing in outputs1" in m.get("message", "") for m in mismatches)

    def test_compare_mixed_missing_keys(self):
        """Test comparing with different missing keys in both outputs."""
        outputs1 = {"Y1": np.array([1.0, 2.0])}
        outputs2 = {"Y2": np.array([3.0, 4.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)

        assert match is False
        assert len(mismatches) == 2  # Both missing_in_2 and missing_in_1
