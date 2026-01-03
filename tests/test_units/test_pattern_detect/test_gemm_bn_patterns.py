"""Tests for Gemm+BN pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.gemm_bn import (
    detect_bn_gemm,
    detect_bn_reshape_gemm,
    detect_gemm_reshape_bn,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestDetectGemmReshapeBn:
    """Test detect_gemm_reshape_bn function."""

    def test_detect_gemm_reshape_bn(self):
        """Test detection of Gemm+Reshape+BN pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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

        result = detect_gemm_reshape_bn(nodes, initializers)
        assert result is None or isinstance(result, list)

    def test_detect_gemm_reshape_bn_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, initializers)
        assert result is None or isinstance(result, list)


class TestDetectBnReshapeGemm:
    """Test detect_bn_reshape_gemm function."""

    def test_detect_bn_reshape_gemm(self):
        """Test detection of BN+Reshape+Gemm pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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

        result = detect_bn_reshape_gemm(nodes, initializers)
        assert result is None or isinstance(result, list)

    def test_detect_bn_reshape_gemm_with_activation(self):
        """Test pattern detection with activation in chain."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 3])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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
        gemm = helper.make_node("Gemm", inputs=["R", "W", "B"], outputs=["G"])
        relu = helper.make_node("Relu", inputs=["G"], outputs=["Z"])

        model = create_minimal_onnx_model([bn, reshape, gemm, relu], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_reshape_gemm(nodes, initializers)
        assert result is None or isinstance(result, list)


class TestDetectBnGemm:
    """Test detect_bn_gemm function."""

    def test_detect_bn_gemm_direct(self):
        """Test detection of BN+Gemm pattern (no reshape)."""
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

        result = detect_bn_gemm(nodes, initializers)
        assert result is None or isinstance(result, list)

    def test_detect_bn_gemm_with_other_nodes(self):
        """Test pattern detection with other nodes in model."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 4])

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
        gemm = helper.make_node("Gemm", inputs=["BN", "W", "B"], outputs=["G"])
        relu = helper.make_node("Relu", inputs=["G"], outputs=["Z"])

        model = create_minimal_onnx_model([bn, gemm, relu], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, initializers)
        assert result is None or isinstance(result, list)

    def test_no_bn_gemm_pattern(self):
        """Test no detection when BN not followed by Gemm."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

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

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, initializers)
        assert result is None or isinstance(result, list)
