"""Extended tests for ONNX utility functions - missing coverage."""

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


class TestConvertConstantToInitializerExtended:
    """Extended tests for convert_constant_to_initializer."""

    def test_converts_float_constant_node_to_initializer(self):
        """Test converting a Constant node with float values to initializer."""
        const_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = numpy_helper.from_array(const_value, "const_out")
        attr = helper.make_attribute("value", tensor_proto)
        const_node = helper.make_node("Constant", inputs=[], outputs=["const_out"])
        const_node.attribute.append(attr)

        nodes = [const_node]
        initializers: dict[str, TensorProto] = {}

        result = convert_constant_to_initializer(nodes, initializers)

        assert len(result) == 0
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

    @pytest.mark.parametrize(
        ("proto_dtype", "np_dtype"),
        [
            pytest.param(TensorProto.FLOAT16, np.float16, id="float16"),
            pytest.param(TensorProto.DOUBLE, np.float64, id="float64"),
            pytest.param(TensorProto.INT32, np.int32, id="int32"),
            pytest.param(TensorProto.INT64, np.int64, id="int64"),
            pytest.param(TensorProto.UINT8, np.uint8, id="uint8"),
            pytest.param(TensorProto.INT8, np.int8, id="int8"),
        ],
    )
    def test_generate_typed_input(self, proto_dtype, np_dtype):
        """Verify generate_random_inputs returns the correct numpy dtype for each ONNX TensorProto type."""
        graph = helper.make_graph(
            [helper.make_node("Identity", inputs=["X"], outputs=["Y"])],
            "test",
            [helper.make_tensor_value_info("X", proto_dtype, [2, 3])],
            [helper.make_tensor_value_info("Y", proto_dtype, [2, 3])],
        )
        model = helper.make_model(graph)

        result = generate_random_inputs(model, num_samples=1)

        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].dtype == np_dtype
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

    @pytest.mark.parametrize(
        ("data", "expected_len"),
        [
            pytest.param(np.array([1.0, 2.0, 3.0], dtype=np.float32), 1, id="1d"),
            pytest.param(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                2,
                id="2d",
            ),
        ],
    )
    def test_loads_npy_arrays(self, data, expected_len, temp_npy_path):
        """Test loading numpy arrays from .npy file."""
        np.save(temp_npy_path, data)

        result = load_test_data_from_file(temp_npy_path)
        assert len(result) == expected_len
        assert "input" in result[0]
        if expected_len == 1:
            assert np.allclose(result[0]["input"], data)
        else:
            assert np.allclose(result[0]["input"], data[0])

    @pytest.mark.parametrize(
        "key_name",
        [
            pytest.param("inputs", id="inputs_key"),
            pytest.param("X", id="x_key"),
            pytest.param("custom_key", id="arbitrary_key"),
        ],
    )
    def test_loads_npz_with_various_keys(self, key_name, temp_npz_path):
        """Test loading .npz files with various key names."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.savez(temp_npz_path, **{key_name: data})

        result = load_test_data_from_file(temp_npz_path)
        assert len(result) == 2
        assert "input" in result[0]
        assert np.allclose(result[0]["input"], data[0])

    def test_raises_on_nonexistent_file(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"Test data file not found"):
            load_test_data_from_file("/nonexistent/path/data.npy")

    def test_raises_on_unsupported_format(self):
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

    @pytest.mark.parametrize(
        ("outputs1", "outputs2", "expected_missing_msg"),
        [
            pytest.param(
                {"Y1": np.array([1.0, 2.0])},
                {},
                "missing in outputs2",
                id="missing_in_outputs2",
            ),
            pytest.param(
                {},
                {"Y1": np.array([1.0, 2.0])},
                "missing in outputs1",
                id="missing_in_outputs1",
            ),
        ],
    )
    def test_detects_missing_outputs(self, outputs1, outputs2, expected_missing_msg):
        """Test detecting missing outputs in comparisons."""
        match, mismatches = compare_outputs(outputs1, outputs2)

        assert match is False
        assert len(mismatches) > 0
        assert any(m.get("type") == "missing_key" for m in mismatches)
        assert any(expected_missing_msg in m.get("message", "") for m in mismatches)

    def test_compare_mixed_missing_keys(self):
        """Test comparing with different missing keys in both outputs."""
        outputs1 = {"Y1": np.array([1.0, 2.0])}
        outputs2 = {"Y2": np.array([3.0, 4.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)

        assert match is False
        assert len(mismatches) == 2  # Both missing_in_2 and missing_in_1
