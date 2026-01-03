"""Tests for ONNX utility functions."""

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.utils import (
    EXTRACT_ATTR_MAP,
    clear_onnx_docstring,
    compare_outputs,
    convert_constant_to_initializer,
    extract_nodes,
    generate_random_inputs,
    get_initializers,
    get_input_nodes,
    get_next_nodes_mapping,
    get_output_nodes,
    reformat_io_shape,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestClearOnnxDocstring:
    """Test clear_onnx_docstring function."""

    def test_clear_docstring_single_node(self):
        """Test clearing docstring from single node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], doc_string="Test doc")
        model = create_minimal_onnx_model([node], [X], [Y])

        result = clear_onnx_docstring(model)
        assert result.graph.node[0].doc_string == ""

    def test_clear_docstring_multiple_nodes(self):
        """Test clearing docstrings from multiple nodes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], doc_string="Doc 1")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], doc_string="Doc 2")
        model = create_minimal_onnx_model([node1, node2], [X], [Z])

        result = clear_onnx_docstring(model)
        assert all(node.doc_string == "" for node in result.graph.node)


class TestReformatIOShape:
    """Test reformat_io_shape function."""

    def test_reformat_shape_with_batch(self):
        """Test reformatting shape with batch dimension."""
        node = create_tensor_value_info("X", "float32", [5, 3])
        result = reformat_io_shape(node, has_batch_dim=True)
        # Batch dimension should be normalized to 1
        assert result[0] == 1
        assert result[1] == 3

    def test_reformat_shape_no_batch_dim(self):
        """Test reformatting shape without batch dimension requirement."""
        node = create_tensor_value_info("X", "float32", [3])
        result = reformat_io_shape(node, has_batch_dim=False)
        assert result == [3]

    def test_reformat_shape_scalar(self):
        """Test reformatting scalar output."""
        node = create_tensor_value_info("X", "float32", [])
        result = reformat_io_shape(node, has_batch_dim=True)
        assert result == []

    def test_reformat_shape_batch_dim_error(self):
        """Test error when batch dimension missing."""
        node = create_tensor_value_info("X", "float32", [3])
        with pytest.raises(ValueError, match="batch dimension"):
            reformat_io_shape(node, has_batch_dim=True)


class TestGetInputOutputNodes:
    """Test get_input_nodes and get_output_nodes functions."""

    def test_get_input_nodes_single(self):
        """Test getting single input node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        initializers = {init.name: init for init in model.graph.initializer}
        result = get_input_nodes(model, initializers, has_batch_dim=False)
        assert len(result) == 1
        assert result[0].name == "X"

    def test_get_output_nodes_single(self):
        """Test getting single output node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        result = get_output_nodes(model, has_batch_dim=False)
        assert len(result) == 1
        assert result[0].name == "Y"


class TestGetInitializers:
    """Test get_initializers function."""

    def test_get_initializers_single(self):
        """Test getting single initializer."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])
        W = create_initializer("W", np.ones((3, 3), dtype=np.float32))

        node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y], [W])

        result = get_initializers(model)
        assert "W" in result
        assert result["W"] == W

    def test_get_initializers_empty(self):
        """Test getting initializers when none exist."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        result = get_initializers(model)
        assert len(result) == 0


class TestConvertConstantToInitializer:
    """Test convert_constant_to_initializer function."""

    def test_convert_no_constant_nodes(self):
        """Test when no constant nodes present."""
        relu_node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        nodes = [relu_node]
        initializers: dict[str, object] = {}

        result = convert_constant_to_initializer(nodes, initializers)
        assert len(result) == 1
        assert result[0].op_type == "Relu"


class TestExtractNodes:
    """Test extract_nodes function."""

    def test_extract_nodes_basic(self):
        """Test extracting nodes from model."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        input_nodes, output_nodes, nodes, _ = extract_nodes(model, has_batch_dim=False)
        assert len(input_nodes) == 1
        assert len(output_nodes) == 1
        assert len(nodes) == 1
        assert input_nodes[0].name == "X"
        assert output_nodes[0].name == "Y"

    def test_extract_nodes_with_initializers(self):
        """Test extracting nodes with initializers."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])
        W = create_initializer("W", np.ones((3, 3), dtype=np.float32))

        node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y], [W])

        input_nodes, _, _, initializers = extract_nodes(model, has_batch_dim=False)
        assert "W" in initializers
        assert input_nodes[0].name == "X"


class TestGetNextNodesMapping:
    """Test get_next_nodes_mapping function."""

    def test_single_node_mapping(self):
        """Test mapping for single node."""
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        result = get_next_nodes_mapping([node])
        assert isinstance(result, dict)
        assert "relu_0" in result

    def test_chained_nodes_mapping(self):
        """Test mapping for chained nodes."""
        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        result = get_next_nodes_mapping([node1, node2])
        # node1's output Y should map to node2
        assert result["relu_0"] == ["relu_1"]

    def test_branching_mapping(self):
        """Test mapping with branching."""
        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")
        node3 = helper.make_node("Add", inputs=["Y", "Y"], outputs=["W"], name="add_0")

        result = get_next_nodes_mapping([node1, node2, node3])
        # node1's output Y is consumed by both node2 and node3
        assert "relu_0" in result
        assert len(result["relu_0"]) >= 1


class TestGenerateRandomInputs:
    """Test generate_random_inputs function."""

    def test_generate_single_input(self):
        """Test generating single input."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        result = generate_random_inputs(model, num_samples=1)
        assert len(result) == 1
        assert "X" in result[0]
        assert result[0]["X"].shape == (2, 3)

    def test_generate_multiple_samples(self):
        """Test generating multiple samples."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([node], [X], [Y])

        result = generate_random_inputs(model, num_samples=3)
        assert len(result) == 3
        for sample in result:
            assert "X" in sample
            assert sample["X"].shape == (2, 3)

    def test_generate_multiple_inputs(self):
        """Test generating with multiple input tensors."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 4])

        node = helper.make_node("MatMul", inputs=["X", "Y"], outputs=["Z"])
        model = create_minimal_onnx_model(
            [node], [X], [Z], [create_initializer("Y", np.ones((3, 4)))]
        )

        result = generate_random_inputs(model, num_samples=1)
        assert len(result) == 1
        assert "X" in result[0]


class TestExtractAttrMap:
    """Test EXTRACT_ATTR_MAP functionality."""

    def test_extract_attr_map_has_entries(self):
        """Test EXTRACT_ATTR_MAP has expected entries."""
        assert 0 in EXTRACT_ATTR_MAP  # UNDEFINED
        assert 1 in EXTRACT_ATTR_MAP  # FLOAT
        assert 2 in EXTRACT_ATTR_MAP  # INT
        assert 3 in EXTRACT_ATTR_MAP  # STRING
        assert 6 in EXTRACT_ATTR_MAP  # FLOATS
        assert 7 in EXTRACT_ATTR_MAP  # INTS

    def test_extract_attr_undefined(self):
        """Test extracting undefined attribute returns None."""
        extractor = EXTRACT_ATTR_MAP[0]
        result = extractor(None)
        assert result is None


class TestCompareOutputs:
    """Test compare_outputs function."""

    def test_compare_identical_outputs(self):
        """Test comparing identical outputs."""
        outputs1 = {"Y": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"Y": np.array([1.0, 2.0, 3.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)
        assert match is True
        assert len(mismatches) == 0

    def test_compare_close_outputs(self):
        """Test comparing close outputs within tolerance."""
        outputs1 = {"Y": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"Y": np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])}

        match, _ = compare_outputs(outputs1, outputs2, rtol=1e-5, atol=1e-6)
        assert match is True

    def test_compare_different_outputs(self):
        """Test comparing different outputs."""
        outputs1 = {"Y": np.array([1.0, 2.0, 3.0])}
        outputs2 = {"Y": np.array([1.0, 2.5, 3.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)
        assert match is False
        assert len(mismatches) > 0

    def test_compare_multiple_outputs(self):
        """Test comparing multiple outputs."""
        outputs1 = {
            "Y1": np.array([1.0, 2.0]),
            "Y2": np.array([3.0, 4.0]),
        }
        outputs2 = {
            "Y1": np.array([1.0, 2.0]),
            "Y2": np.array([3.0, 4.0]),
        }

        match, _ = compare_outputs(outputs1, outputs2)
        assert match is True

    def test_compare_missing_output(self):
        """Test comparing when output missing."""
        outputs1 = {"Y": np.array([1.0])}
        outputs2: dict[str, np.ndarray] = {}

        match, mismatches = compare_outputs(outputs1, outputs2)
        assert match is False
        assert len(mismatches) > 0

    def test_compare_output_shape_mismatch(self):
        """Test comparing with mismatched shapes."""
        outputs1 = {"Y": np.array([1.0, 2.0])}
        outputs2 = {"Y": np.array([1.0, 2.0, 3.0])}

        match, mismatches = compare_outputs(outputs1, outputs2)
        assert match is False
        assert any(m.get("type") == "shape_mismatch" for m in mismatches)
