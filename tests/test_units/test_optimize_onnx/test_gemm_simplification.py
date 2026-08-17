"""Tests for Gemm simplification and normalization."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._gemm import (
    _normalize_gemm_bias_input,
    _normalize_gemm_matrix_input,
    _simplify_gemm,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestNormalizeGemmMatrixInput:
    """Test _normalize_gemm_matrix_input function."""

    @pytest.mark.parametrize(
        ("scale_factor", "should_transpose"),
        [(1.0, False), (2.0, False), (1.0, True)],
    )
    def test_normalizes_matrix_input(self, scale_factor, should_transpose):
        """Test normalization with various scale and transpose combinations."""
        A_array = np.random.randn(2, 3).astype(np.float32)
        A = create_initializer("A", A_array)
        initializers = {"A": A}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "A",
            scale_factor,
            should_transpose=should_transpose,
            initializers=initializers,
            unique_suffix="0",
        )

        # When input is an initializer, always creates a copy
        assert input_name == "A_0"
        assert scale == 1.0
        assert trans == 0
        assert "A_0" in initializers

    @pytest.mark.parametrize("is_initializer", [True, False])
    def test_normalizes_initializer_vs_variable(self, is_initializer):
        """Test normalization behavior with initializers vs variables."""
        initializers: dict[str, Any] = {}
        scale_factor = 0.5
        should_transpose = True

        if is_initializer:
            A_array = np.ones((2, 3), dtype=np.float32) * 2.0
            A = create_initializer("A", A_array)
            initializers["A"] = A
            input_name, scale, trans = _normalize_gemm_matrix_input(
                "A",
                scale_factor,
                should_transpose=should_transpose,
                initializers=initializers,
                unique_suffix="0",
            )
            # Should create new initializer with transposed and scaled values
            assert input_name == "A_0"
            assert scale == 1.0
            assert trans == 0
            assert "A_0" in initializers
        else:
            # Non-initializer input
            input_name, scale, trans = _normalize_gemm_matrix_input(
                "X",
                scale_factor,
                should_transpose=should_transpose,
                initializers=initializers,
                unique_suffix="0",
            )
            # Should return modified transpose flag
            assert input_name == "X"
            assert scale == 0.5
            assert trans == 1


class TestNormalizeGemmBiasInput:
    """Test _normalize_gemm_bias_input function."""

    def test_normalizes_with_beta_one(self):
        """Test bias normalization with beta=1."""
        C_array = np.random.randn(3).astype(np.float32)
        C = create_initializer("C", C_array)
        initializers = {"C": C}

        input_name, beta = _normalize_gemm_bias_input("C", 1.0, initializers, "0")

        # When input is an initializer, always creates a copy
        assert input_name == "C_0"
        assert beta == 1.0
        assert "C_0" in initializers

    def test_scales_bias_correctly(self):
        """Test bias normalization with beta scaling."""
        C_array = np.ones(3, dtype=np.float32) * 2.0
        C = create_initializer("C", C_array)
        initializers = {"C": C}

        input_name, beta = _normalize_gemm_bias_input("C", 0.5, initializers, "0")

        # Should create new initializer with scaled values
        assert input_name == "C_0"
        assert beta == 1.0
        assert "C_0" in initializers

    def test_handles_non_initializer_bias(self):
        """Test bias normalization with non-initializer input."""
        initializers: dict[str, Any] = {}

        input_name, beta = _normalize_gemm_bias_input("C", 2.0, initializers, "0")

        # Should return unchanged
        assert input_name == "C"
        assert beta == 2.0


class TestSimplifyGemm:
    """Test _simplify_gemm function."""

    def test_simplifies_single_node(self):
        """Test simplifying a single Gemm node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W_array = np.random.randn(3, 4).astype(np.float32)
        B_array = np.random.randn(4).astype(np.float32)

        initializers_list = [
            create_initializer("W", W_array),
            create_initializer("B", B_array),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            alpha=1.0,
            beta=1.0,
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 1
        assert result[0].op_type == "Gemm"

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [(2.0, 1.0), (1.0, 0.5)],
    )
    def test_handles_scaling_parameters(self, alpha, beta):
        """Test simplifying Gemm with alpha and beta scaling."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W_array = np.ones((3, 4), dtype=np.float32)
        B_array = np.ones(4, dtype=np.float32) * 2.0

        initializers_list = [
            create_initializer("W", W_array),
            create_initializer("B", B_array),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            alpha=alpha,
            beta=beta,
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 1
        # Scaled values should be absorbed into initializers
        assert "W" in initializers_dict or "W_0" in initializers_dict
        assert "B" in initializers_dict or "B_0" in initializers_dict

    def test_simplifies_without_bias(self):
        """Test simplifying Gemm without bias input."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W_array = np.random.randn(3, 4).astype(np.float32)
        initializers_list = [create_initializer("W", W_array)]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "W"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 1
        assert len(result[0].input) == 2  # Only X and W

    def test_simplifies_multiple_nodes(self):
        """Test simplifying multiple Gemm nodes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1_array = np.random.randn(3, 4).astype(np.float32)
        B1_array = np.random.randn(4).astype(np.float32)
        W2_array = np.random.randn(4, 5).astype(np.float32)
        B2_array = np.random.randn(5).astype(np.float32)

        initializers_list = [
            create_initializer("W1", W1_array),
            create_initializer("B1", B1_array),
            create_initializer("W2", W2_array),
            create_initializer("B2", B2_array),
        ]

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 2
        assert result[0].op_type == "Gemm"
        assert result[1].op_type == "Gemm"

    def test_removes_unused_initializers_and_simplifies_graph(self):
        """Test that simplify removes unused initializers and preserves graph structure."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W_array = np.random.randn(3, 4).astype(np.float32)
        unused_array = np.random.randn(5, 6).astype(np.float32)

        initializers_list = [
            create_initializer("W", W_array),
            create_initializer("unused", unused_array),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "W"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        initial_count = len(initializers_dict)
        result = _simplify_gemm(nodes, initializers_dict)

        # Unused initializer should be removed
        assert len(initializers_dict) < initial_count
        # Gemm node should be preserved
        assert len(result) == 1
        assert result[0].op_type == "Gemm"

    def test_preserves_non_gemm_nodes(self):
        """Test that simplify preserves non-Gemm nodes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 4])

        W_array = np.random.randn(3, 4).astype(np.float32)
        initializers_list = [create_initializer("W", W_array)]

        gemm_node = helper.make_node("Gemm", inputs=["X", "W"], outputs=["Y"])
        relu_node = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm_node, relu_node], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 2
        assert result[0].op_type == "Gemm"
        assert result[1].op_type == "Relu"

    def test_preserves_matrix_operand_order(self):
        """A constant left operand is not swapped with a runtime right operand."""
        a = create_initializer("A", np.arange(6, dtype=np.float32).reshape(2, 3))
        node = helper.make_node("Gemm", inputs=["A", "X"], outputs=["Y"])
        initializers = {"A": a}

        result = _simplify_gemm([node], initializers)

        assert list(result[0].input) == ["A_0", "X"]

    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            ({"transA": 1}, {"transA": 1}),
            ({"transB": 1}, {"transB": 1}),
            ({"alpha": 2.0}, {"alpha": 2.0}),
        ],
        ids=["dynamic_transa", "dynamic_transb", "dynamic_alpha"],
    )
    def test_preserves_attributes_on_dynamic_matrix_inputs(self, attrs, expected):
        """Attributes remain on the node when no initializer can absorb them."""
        node = helper.make_node("Gemm", inputs=["X", "W"], outputs=["Y"], **attrs)

        result = _simplify_gemm([node], {})

        actual = {attr.name: helper.get_attribute_value(attr) for attr in result[0].attribute}
        assert actual == expected

    def test_preserves_beta_on_dynamic_bias(self):
        """A runtime C input keeps its beta scale on the normalized node."""
        weight = create_initializer("W", np.eye(2, dtype=np.float32))
        node = helper.make_node("Gemm", inputs=["X", "W", "C"], outputs=["Y"], beta=0.25)

        result = _simplify_gemm([node], {"W": weight})

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in result[0].attribute}
        assert attrs == {"beta": pytest.approx(0.25)}
