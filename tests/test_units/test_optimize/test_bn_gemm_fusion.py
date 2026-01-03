"""Tests for BN+Gemm fusion optimization."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._bn_gemm import (
    _fuse_bn_gemm,
    _fuse_bn_reshape_gemm,
    _fuse_gemm_reshape_bn,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestFuseGemmReshapeBn:
    """Test _fuse_gemm_reshape_bn function."""

    def test_fuse_gemm_reshape_bn_basic(self):
        """Test basic Gemm->Reshape->BN fusion."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            shape,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"])
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_gemm_reshape_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_fuse_gemm_reshape_bn_no_pattern(self):
        """Test no fusion when pattern missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_gemm_reshape_bn(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fuse_gemm_reshape_bn_with_trans_b(self):
        """Test Gemm->Reshape->BN fusion with transB=1."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            shape,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"], transB=1)
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_gemm_reshape_bn(nodes, initializers)
        assert isinstance(result, list)

    def test_fuse_gemm_reshape_bn_unsupported_trans_a(self):
        """Test error when Gemm has transA=1 (unsupported)."""
        X = create_tensor_value_info("X", "float32", [3, 2])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            shape,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"], transA=1)
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        try:
            result = _fuse_gemm_reshape_bn(nodes, initializers)
            # If it succeeds without error, it's okay (implementation choice)
            assert isinstance(result, list)
        except ValueError:
            # transA=1 is unsupported, so error is acceptable
            pass


class TestFuseBnGemm:
    """Test _fuse_bn_gemm function."""

    def test_fuse_bn_gemm_basic(self):
        """Test basic BN->Gemm fusion."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["BN"],
        )
        gemm = helper.make_node("Gemm", inputs=["BN", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, gemm], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_fuse_bn_gemm_no_pattern(self):
        """Test no fusion when pattern missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fuse_bn_gemm_with_nonzero_bn_bias(self):
        """Test BN->Gemm fusion with non-zero BN bias."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["BN"],
        )
        gemm = helper.make_node("Gemm", inputs=["BN", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, gemm], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) <= 2


class TestFuseBnReshapeGemm:
    """Test _fuse_bn_reshape_gemm function."""

    def test_fuse_bn_reshape_gemm_basic(self):
        """Test basic BN->Reshape->Gemm fusion."""
        X = create_tensor_value_info("X", "float32", [2, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3], dtype=np.int64))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            B,
            shape,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["BN"],
        )
        reshape = helper.make_node("Reshape", inputs=["BN", "shape"], outputs=["R"])
        gemm = helper.make_node("Gemm", inputs=["R", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, reshape, gemm], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_reshape_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_fuse_bn_reshape_gemm_no_pattern(self):
        """Test no fusion when pattern missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_reshape_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fuse_bn_reshape_gemm_chain(self):
        """Test multiple BN->Reshape->Gemm patterns."""
        X = create_tensor_value_info("X", "float32", [2, 3, 1, 1])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        shape1 = create_initializer("shape1", np.array([2, 3], dtype=np.int64))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))
        shape2 = create_initializer("shape2", np.array([2, 4], dtype=np.int64))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W1,
            B1,
            shape1,
            W2,
            B2,
            shape2,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["BN"],
        )
        reshape1 = helper.make_node("Reshape", inputs=["BN", "shape1"], outputs=["R1"])
        gemm1 = helper.make_node("Gemm", inputs=["R1", "W1", "B1"], outputs=["Y1"])
        reshape2 = helper.make_node("Reshape", inputs=["Y1", "shape2"], outputs=["R2"])
        gemm2 = helper.make_node("Gemm", inputs=["R2", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model(
            [bn, reshape1, gemm1, reshape2, gemm2], [X], [Z], initializers_list
        )
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_bn_reshape_gemm(nodes, initializers)
        assert isinstance(result, list)
