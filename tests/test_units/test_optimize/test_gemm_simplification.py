"""Tests for Gemm simplification and normalization."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._gemm import (
    _normalize_gemm_bias_input,
    _normalize_gemm_matrix_input,
    _simplify_gemm,
    _swap_gemm_inputs_if_needed,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestNormalizeGemmMatrixInput:
    """Test _normalize_gemm_matrix_input function."""

    def test_normalize_without_scale_no_transpose(self):
        """Test normalization without scaling and transpose."""
        A = create_initializer("A", np.random.randn(2, 3).astype(np.float32))
        initializers = {"A": A}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "A", 1.0, should_transpose=False, initializers=initializers, unique_suffix="0"
        )

        # When input is an initializer, always creates a copy
        assert input_name == "A_0"
        assert scale == 1.0
        assert trans == 0
        assert "A_0" in initializers

    def test_normalize_with_scale(self):
        """Test normalization with scaling factor."""
        A_array = np.random.randn(2, 3).astype(np.float32)
        A = create_initializer("A", A_array)
        initializers = {"A": A}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "A", 2.0, should_transpose=False, initializers=initializers, unique_suffix="0"
        )

        # Should create new initializer with scaled values
        assert input_name == "A_0"
        assert scale == 1.0
        assert trans == 0
        assert "A_0" in initializers

    def test_normalize_with_transpose(self):
        """Test normalization with transpose."""
        A_array = np.random.randn(2, 3).astype(np.float32)
        A = create_initializer("A", A_array)
        initializers = {"A": A}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "A", 1.0, should_transpose=True, initializers=initializers, unique_suffix="0"
        )

        # Should create new initializer with transposed values
        assert input_name == "A_0"
        assert scale == 1.0
        assert trans == 0
        assert "A_0" in initializers

    def test_normalize_with_scale_and_transpose(self):
        """Test normalization with both scaling and transpose."""
        A_array = np.ones((2, 3), dtype=np.float32) * 2.0
        A = create_initializer("A", A_array)
        initializers = {"A": A}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "A", 0.5, should_transpose=True, initializers=initializers, unique_suffix="0"
        )

        # Should create new initializer with transposed and scaled values
        assert input_name == "A_0"
        assert scale == 1.0
        assert trans == 0
        assert "A_0" in initializers

    def test_normalize_non_initializer_input(self):
        """Test normalization with non-initializer input."""
        initializers: dict[str, Any] = {}

        input_name, scale, trans = _normalize_gemm_matrix_input(
            "X", 1.5, should_transpose=True, initializers=initializers, unique_suffix="0"
        )

        # Should return modified transpose flag
        assert input_name == "X"
        assert scale == 1.5
        assert trans == 1


class TestNormalizeGemmBiasInput:
    """Test _normalize_gemm_bias_input function."""

    def test_normalize_bias_with_beta_one(self):
        """Test bias normalization with beta=1."""
        C_array = np.random.randn(3).astype(np.float32)
        C = create_initializer("C", C_array)
        initializers = {"C": C}

        input_name, beta = _normalize_gemm_bias_input("C", 1.0, initializers, "0")

        # When input is an initializer, always creates a copy
        assert input_name == "C_0"
        assert beta == 1.0
        assert "C_0" in initializers

    def test_normalize_bias_with_beta_scaling(self):
        """Test bias normalization with beta scaling."""
        C_array = np.ones(3, dtype=np.float32) * 2.0
        C = create_initializer("C", C_array)
        initializers = {"C": C}

        input_name, beta = _normalize_gemm_bias_input("C", 0.5, initializers, "0")

        # Should create new initializer with scaled values
        assert input_name == "C_0"
        assert beta == 1.0
        assert "C_0" in initializers

    def test_normalize_bias_non_initializer(self):
        """Test bias normalization with non-initializer input."""
        initializers: dict[str, Any] = {}

        input_name, beta = _normalize_gemm_bias_input("C", 2.0, initializers, "0")

        # Should return unchanged
        assert input_name == "C"
        assert beta == 2.0


class TestSwapGemmInputs:
    """Test _swap_gemm_inputs_if_needed function."""

    def test_no_swap_both_initializers(self):
        """Test no swap when both are initializers."""
        A = create_initializer("A", np.random.randn(2, 3).astype(np.float32))
        B = create_initializer("B", np.random.randn(3, 4).astype(np.float32))
        initializers = {"A": A, "B": B}

        var_name, weight_name = _swap_gemm_inputs_if_needed("A", "B", initializers)

        # Should keep original order
        assert var_name == "A"
        assert weight_name == "B"

    def test_no_swap_both_variables(self):
        """Test no swap when both are variables."""
        initializers: dict[str, Any] = {}

        var_name, weight_name = _swap_gemm_inputs_if_needed("X", "Y", initializers)

        # Should keep original order
        assert var_name == "X"
        assert weight_name == "Y"

    def test_swap_when_first_is_initializer(self):
        """Test swapping when first input is initializer, second is variable."""
        A = create_initializer("A", np.random.randn(2, 3).astype(np.float32))
        initializers = {"A": A}

        var_name, weight_name = _swap_gemm_inputs_if_needed("A", "X", initializers)

        # Should swap
        assert var_name == "X"
        assert weight_name == "A"

    def test_no_swap_first_is_variable(self):
        """Test no swap when first is variable, second is initializer."""
        B = create_initializer("B", np.random.randn(3, 4).astype(np.float32))
        initializers = {"B": B}

        var_name, weight_name = _swap_gemm_inputs_if_needed("X", "B", initializers)

        # Should keep original order
        assert var_name == "X"
        assert weight_name == "B"


class TestSimplifyGemm:
    """Test _simplify_gemm function."""

    def test_simplify_single_gemm(self):
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

    def test_simplify_gemm_with_alpha(self):
        """Test simplifying Gemm with alpha scaling."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W_array = np.ones((3, 4), dtype=np.float32)
        B_array = np.ones(4, dtype=np.float32)

        initializers_list = [
            create_initializer("W", W_array),
            create_initializer("B", B_array),
        ]

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            alpha=2.0,
            beta=1.0,
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 1
        # Weight should be scaled into initializer
        assert "W" in initializers_dict or "W_0" in initializers_dict

    def test_simplify_gemm_with_beta(self):
        """Test simplifying Gemm with beta scaling on bias."""
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
            alpha=1.0,
            beta=0.5,
        )

        model = create_minimal_onnx_model([gemm_node], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _simplify_gemm(nodes, initializers_dict)

        assert len(result) == 1
        # Bias should be scaled into initializer
        assert "B" in initializers_dict or "B_0" in initializers_dict

    def test_simplify_gemm_without_bias(self):
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

    def test_simplify_multiple_gemmnodes(self):
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

    def test_simplify_removes_unused_initializers(self):
        """Test that simplify removes unused initializers."""
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
        _simplify_gemm(nodes, initializers_dict)

        # Unused initializer should be removed
        assert len(initializers_dict) < initial_count

    def test_simplify_preserves_non_gemm_nodes(self):
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
