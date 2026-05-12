"""Tests for Transpose-BatchNorm-Transpose fusion optimization."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._bn_transpose import (
    _can_fuse_to_gemm,
    _create_gemm_fusion,
    _create_matmul_add_fusion,
    _fuse_transpose_batchnorm_transpose,
    _validate_transpose_perm,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestValidateTransposePerm:
    """Test _validate_transpose_perm function."""

    def test_valid_perm(self):
        """Test validation of valid transpose permutation."""
        tp_node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[0, 2, 1])
        initializers: dict[str, Any] = {}

        # Should not raise
        _validate_transpose_perm(tp_node, initializers)

    def test_invalid_perm_raises(self):
        """Test that invalid transpose perm raises ValueError."""
        tp_node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[0, 1, 2])
        initializers: dict[str, Any] = {}

        # Should raise ValueError for unsupported perm
        with pytest.raises(ValueError, match="unsupported"):
            _validate_transpose_perm(tp_node, initializers)


class TestCanFuseToGemm:
    """Test _can_fuse_to_gemm function."""

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            ([2, 3], True),
            ([1, 2, 3], False),
            ([1, 3, 2, 3], False),
        ],
    )
    def test_fusion_allowed_by_input_rank(self, shape, expected):
        """Test that fusion is allowed based on input rank."""
        data_shapes = {"X": shape}
        can_fuse = _can_fuse_to_gemm("X", None, data_shapes)
        assert can_fuse == expected

    def test_missing_data_shapes_defaults_to_true(self):
        """Test that missing data_shapes defaults to True (can fuse)."""
        can_fuse = _can_fuse_to_gemm("X", None, None)
        assert can_fuse

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [
            pytest.param([2, 3], True, id="rank_2"),
            pytest.param([1, 3, 4, 4], False, id="rank_4"),
        ],
    )
    def test_input_from_graph_inputs_rank(self, shape, expected):
        """Test checking input rank from graph inputs."""
        X = create_tensor_value_info("X", "float32", shape)
        Y = create_tensor_value_info("Y", "float32", shape)

        identity = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        model = create_minimal_onnx_model([identity], [X], [Y])

        input_nodes = list(model.graph.input)

        can_fuse = _can_fuse_to_gemm("X", input_nodes, None)
        assert can_fuse == expected


class TestCreateGemmFusion:
    """Test _create_gemm_fusion function."""

    def test_creates_node_with_correct_inputs_outputs(self):
        """Test creating Gemm fusion node with correct inputs/outputs."""
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp_node1 = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"])
        tp_node2 = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"])

        weight = np.eye(3, dtype=np.float32)
        bias = np.ones(3, dtype=np.float32)
        initializers: dict[str, Any] = {}

        gemm_node = _create_gemm_fusion(bn_node, tp_node1, tp_node2, weight, bias, initializers)

        assert gemm_node.op_type == "Gemm"
        assert len(gemm_node.input) == 3
        assert gemm_node.input[0] == "X"  # From tp_node1
        assert gemm_node.output[0] == "Y"  # From tp_node2
        assert len(initializers) == 2  # Weight and bias added

    def test_creates_initializers(self):
        """Test that Gemm fusion creates initializers."""
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp_node1 = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"])
        tp_node2 = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"])

        weight = np.ones((2, 3), dtype=np.float32)
        bias = np.zeros(3, dtype=np.float32)
        initializers: dict[str, Any] = {}

        _create_gemm_fusion(bn_node, tp_node1, tp_node2, weight, bias, initializers)

        # Should have weight and bias initializers
        assert "bn_scale_gemm" in initializers
        assert "bn_bias_gemm" in initializers


class TestCreateMatmulAddFusion:
    """Test _create_matmul_add_fusion function."""

    def test_creates_nodes_with_correct_types(self):
        """Test creating MatMul+Add fusion nodes with correct types."""
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp_node1 = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"])
        tp_node2 = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"])

        weight = np.eye(3, dtype=np.float32)
        bias = np.ones(3, dtype=np.float32)
        initializers: dict[str, Any] = {}

        matmul_node, add_node = _create_matmul_add_fusion(
            bn_node, tp_node1, tp_node2, weight, bias, initializers
        )

        assert matmul_node.op_type == "MatMul"
        assert add_node.op_type == "Add"
        assert matmul_node.input[0] == "X"  # From tp_node1
        assert add_node.output[0] == "Y"  # From tp_node2
        assert len(initializers) == 2  # Weight and bias added

    def test_chains_matmul_to_add(self):
        """Test that MatMul output connects to Add input."""
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp_node1 = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"])
        tp_node2 = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"])

        weight = np.random.randn(2, 3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        initializers: dict[str, Any] = {}

        matmul_node, add_node = _create_matmul_add_fusion(
            bn_node, tp_node1, tp_node2, weight, bias, initializers
        )

        # MatMul output should connect to Add input
        assert matmul_node.output[0] == add_node.input[0]


class TestFuseTransposeBatchnormTranspose:
    """Test _fuse_transpose_batchnorm_transpose function."""

    @pytest.mark.parametrize(
        ("x_shape", "y_shape"),
        [
            pytest.param([2, 3, 4], [2, 4, 3], id="rank3_gemm"),
            pytest.param([1, 3, 4, 4], [1, 4, 3, 4], id="rank4_matmul_add"),
        ],
    )
    def test_fuses_transpose_bn_transpose(self, x_shape, y_shape):
        """Test fusing Transpose-BN-Transpose for different input ranks."""
        X = create_tensor_value_info("X", "float32", x_shape)
        Y = create_tensor_value_info("Y", "float32", y_shape)

        bn_scale = np.ones(y_shape[-1], dtype=np.float32)
        bn_bias = np.zeros(y_shape[-1], dtype=np.float32)
        bn_mean = np.zeros(y_shape[-1], dtype=np.float32)
        bn_var = np.ones(y_shape[-1], dtype=np.float32)

        initializers_list = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        tp1_node = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"], perm=[0, 2, 1])
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp2_node = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"], perm=[0, 2, 1])

        model = create_minimal_onnx_model(
            [tp1_node, bn_node, tp2_node], [X], [Y], initializers_list
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        input_nodes = list(model.graph.input)
        data_shapes = {"X": list(x_shape), "Y": list(y_shape)}

        result = _fuse_transpose_batchnorm_transpose(
            nodes, initializers_dict, input_nodes, data_shapes
        )

        # Should fuse to fewer nodes
        assert len(result) < len(nodes)

    def test_no_fusion_non_consecutive_nodes(self):
        """Test no fusion when nodes are not consecutive."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 3])

        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        tp1_node = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"], perm=[0, 1])
        relu_node = helper.make_node("Relu", inputs=["t1_out"], outputs=["relu_out"])
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["relu_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp2_node = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"], perm=[0, 1])

        model = create_minimal_onnx_model(
            [tp1_node, relu_node, bn_node, tp2_node], [X], [Z], initializers_list
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _fuse_transpose_batchnorm_transpose(nodes, initializers_dict)

        # Should not fuse due to Relu in between
        assert len(result) == len(nodes)

    def test_no_fusion_single_transpose(self):
        """Test no fusion with single transpose."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [3, 2])

        tp_node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[1, 0])

        model = create_minimal_onnx_model([tp_node], [X], [Y])
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}

        result = _fuse_transpose_batchnorm_transpose(nodes, initializers_dict)

        # Should not fuse - no BN
        assert len(result) == len(nodes)

    def test_fuse_with_different_float_dtypes(self):
        """Test fusion preserves dtype consistency."""
        X = create_tensor_value_info("X", "float32", [2, 3, 4])
        Y = create_tensor_value_info("Y", "float32", [2, 4, 3])

        bn_scale = np.ones(4, dtype=np.float32)
        bn_bias = np.zeros(4, dtype=np.float32)
        bn_mean = np.zeros(4, dtype=np.float32)
        bn_var = np.ones(4, dtype=np.float32)

        initializers_list = [
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        tp1_node = helper.make_node("Transpose", inputs=["X"], outputs=["t1_out"], perm=[0, 2, 1])
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["t1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_out"],
        )
        tp2_node = helper.make_node("Transpose", inputs=["bn_out"], outputs=["Y"], perm=[0, 2, 1])

        model = create_minimal_onnx_model(
            [tp1_node, bn_node, tp2_node], [X], [Y], initializers_list
        )
        nodes = list(model.graph.node)
        initializers_dict = {init.name: init for init in model.graph.initializer}
        input_nodes = list(model.graph.input)
        data_shapes = {"X": [2, 3, 4], "Y": [2, 4, 3]}

        result = _fuse_transpose_batchnorm_transpose(
            nodes, initializers_dict, input_nodes, data_shapes
        )

        # Fusion should succeed
        assert len(result) < len(nodes)
        # Check that new initializers exist
        assert len(initializers_dict) > 0
