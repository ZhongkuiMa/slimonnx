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
from slimonnx.pattern_detect.utils import is_consecutive_nodes as _is_consecutive_nodes

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestIsConsecutiveNodes:
    """Test _is_consecutive_nodes helper function."""

    def test_is_consecutive_nodes_true(self):
        """Test nodes that are consecutive."""
        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        nodes = [node1, node2]

        assert _is_consecutive_nodes(node1, node2, nodes) is True

    def test_is_consecutive_nodes_branching(self):
        """Test nodes with branching (multiple consumers)."""
        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
        node3 = helper.make_node("Relu", inputs=["Y"], outputs=["W"])
        nodes = [node1, node2, node3]

        assert _is_consecutive_nodes(node1, node2, nodes) is False

    def test_is_consecutive_nodes_wrong_input_output(self):
        """Test nodes with mismatched inputs/outputs."""
        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node2 = helper.make_node("Relu", inputs=["Z"], outputs=["W"])
        nodes = [node1, node2]

        assert _is_consecutive_nodes(node1, node2, nodes) is False


class TestDetectGemmReshapeBn:
    """Test detect_gemm_reshape_bn pattern detection."""

    def test_detect_gemm_reshape_bn_valid_pattern(self):
        """Test detecting valid Gemm->Reshape->BN pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"])
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        initializers = [W, B, shape, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, init_dict)
        assert len(result) == 1
        assert result[0]["gemm_node"] == gemm.name
        assert result[0]["reshape_node"] == reshape.name
        assert result[0]["bn_node"] == bn.name

    def test_detect_gemm_reshape_bn_missing_weights(self):
        """Test no detection when Gemm has missing weight."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X"], outputs=["G"])  # Missing W, B
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        initializers = [shape, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, init_dict)
        assert len(result) == 0

    def test_detect_gemm_reshape_bn_missing_reshape_shape(self):
        """Test no detection when Reshape has missing shape."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"])
        reshape = helper.make_node("Reshape", inputs=["G"], outputs=["R"])  # Missing shape
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        initializers = [W, B, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, init_dict)
        assert len(result) == 0

    def test_detect_gemm_reshape_bn_missing_bn_params(self):
        """Test no detection when BN has missing parameters."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"])
        reshape = helper.make_node("Reshape", inputs=["G", "shape"], outputs=["R"])
        bn = helper.make_node("BatchNormalization", inputs=["R"], outputs=["Y"])  # Missing params

        initializers = [W, B, shape]
        model = create_minimal_onnx_model([gemm, reshape, bn], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, init_dict)
        assert len(result) == 0

    def test_detect_gemm_reshape_bn_not_consecutive(self):
        """Test no detection when nodes are not consecutive."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3, 1, 1])

        W = create_initializer("W", np.random.randn(3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3, 1, 1], dtype=np.int64))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["G"])
        relu = helper.make_node("Relu", inputs=["G"], outputs=["R_relu"])
        reshape = helper.make_node("Reshape", inputs=["R_relu", "shape"], outputs=["R"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["R", "scale", "bias", "mean", "var"],
            outputs=["Y"],
        )

        initializers = [W, B, shape, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([gemm, relu, reshape, bn], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_reshape_bn(nodes, init_dict)
        assert len(result) == 0


class TestDetectBnReshapeGemm:
    """Test detect_bn_reshape_gemm pattern detection."""

    def test_detect_bn_reshape_gemm_valid_pattern(self):
        """Test detecting valid BN->Reshape->Gemm pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3], dtype=np.int64))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["BN"],
        )
        reshape = helper.make_node("Reshape", inputs=["BN", "shape"], outputs=["R"])
        gemm = helper.make_node("Gemm", inputs=["R", "W", "B"], outputs=["Y"])

        initializers = [W, B, shape, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([bn, reshape, gemm], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_reshape_gemm(nodes, init_dict)
        assert len(result) == 1
        assert result[0]["bn_node"] == bn.name

    def test_detect_bn_reshape_gemm_missing_bn_params(self):
        """Test no detection when BN has missing parameters."""
        X = create_tensor_value_info("X", "float32", [2, 3, 1, 1])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        shape = create_initializer("shape", np.array([2, 3], dtype=np.int64))

        bn = helper.make_node("BatchNormalization", inputs=["X"], outputs=["BN"])
        reshape = helper.make_node("Reshape", inputs=["BN", "shape"], outputs=["R"])
        gemm = helper.make_node("Gemm", inputs=["R", "W", "B"], outputs=["Y"])

        initializers = [W, B, shape]
        model = create_minimal_onnx_model([bn, reshape, gemm], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_reshape_gemm(nodes, init_dict)
        assert len(result) == 0


class TestDetectBnGemm:
    """Test detect_bn_gemm pattern detection."""

    def test_detect_bn_gemm_valid_pattern(self):
        """Test detecting valid BN->Gemm pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["BN"],
        )
        gemm = helper.make_node("Gemm", inputs=["BN", "W", "B"], outputs=["Y"])

        initializers = [W, B, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([bn, gemm], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, init_dict)
        assert len(result) == 1
        assert result[0]["bn_node"] == bn.name
        assert result[0]["gemm_node"] == gemm.name

    def test_detect_bn_gemm_multiple_instances(self):
        """Test detecting multiple BN->Gemm patterns."""
        X1 = create_tensor_value_info("X1", "float32", [2, 3])
        Y2 = create_tensor_value_info("Y2", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        bn_scale1 = create_initializer("scale1", np.ones(3, dtype=np.float32))
        bn_bias1 = create_initializer("bias1", np.zeros(3, dtype=np.float32))
        bn_mean1 = create_initializer("mean1", np.zeros(3, dtype=np.float32))
        bn_var1 = create_initializer("var1", np.ones(3, dtype=np.float32))

        bn_scale2 = create_initializer("scale2", np.ones(4, dtype=np.float32))
        bn_bias2 = create_initializer("bias2", np.zeros(4, dtype=np.float32))
        bn_mean2 = create_initializer("mean2", np.zeros(4, dtype=np.float32))
        bn_var2 = create_initializer("var2", np.ones(4, dtype=np.float32))

        bn1 = helper.make_node(
            "BatchNormalization",
            inputs=["X1", "scale1", "bias1", "mean1", "var1"],
            outputs=["BN1"],
        )
        gemm1 = helper.make_node("Gemm", inputs=["BN1", "W1", "B1"], outputs=["Y1"])
        bn2 = helper.make_node(
            "BatchNormalization",
            inputs=["Y1", "scale2", "bias2", "mean2", "var2"],
            outputs=["BN2"],
        )
        gemm2 = helper.make_node("Gemm", inputs=["BN2", "W2", "B2"], outputs=["Y2"])

        initializers = [
            W1,
            B1,
            W2,
            B2,
            bn_scale1,
            bn_bias1,
            bn_mean1,
            bn_var1,
            bn_scale2,
            bn_bias2,
            bn_mean2,
            bn_var2,
        ]
        model = create_minimal_onnx_model([bn1, gemm1, bn2, gemm2], [X1], [Y2], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, init_dict)
        assert len(result) == 2

    def test_detect_bn_gemm_missing_gemm_weight(self):
        """Test no detection when Gemm has missing weight."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["BN"],
        )
        gemm = helper.make_node("Gemm", inputs=["BN"], outputs=["Y"])  # Missing W, B

        initializers = [bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([bn, gemm], [X], [Y], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, init_dict)
        assert len(result) == 0

    def test_detect_bn_gemm_branching(self):
        """Test no detection when BN output has multiple consumers."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y1 = create_tensor_value_info("Y1", "float32", [2, 4])
        Y2 = create_tensor_value_info("Y2", "float32", [2, 3])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))
        bn_scale = create_initializer("scale", np.ones(3, dtype=np.float32))
        bn_bias = create_initializer("bias", np.zeros(3, dtype=np.float32))
        bn_mean = create_initializer("mean", np.zeros(3, dtype=np.float32))
        bn_var = create_initializer("var", np.ones(3, dtype=np.float32))

        bn = helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["BN"],
        )
        gemm = helper.make_node("Gemm", inputs=["BN", "W", "B"], outputs=["Y1"])
        relu = helper.make_node("Relu", inputs=["BN"], outputs=["Y2"])

        initializers = [W, B, bn_scale, bn_bias, bn_mean, bn_var]
        model = create_minimal_onnx_model([bn, gemm, relu], [X], [Y1, Y2], initializers)
        nodes = list(model.graph.node)
        init_dict = {init.name: init for init in model.graph.initializer}

        result = detect_bn_gemm(nodes, init_dict)
        assert len(result) == 0
