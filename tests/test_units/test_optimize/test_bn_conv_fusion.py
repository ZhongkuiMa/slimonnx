"""Tests for Conv+BN fusion optimization implementation."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._bn_conv import (
    _fuse_conv_bn_or_bn_conv,
    _fuse_conv_transpose_bn_or_bn_conv_transpose,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestFuseConvBn:
    """Test Conv+BN fusion implementation."""

    def test_fuse_conv_bn_basic(self):
        """Test basic Conv->BN fusion."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Conv"
        assert len(result[0].input) == 3  # input, weight, bias

    def test_fuse_conv_bn_without_bias(self):
        """Test Conv->BN fusion when Conv has no bias."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        bn_scale = np.ones(3, dtype=np.float32)
        bn_bias = np.zeros(3, dtype=np.float32)
        bn_mean = np.zeros(3, dtype=np.float32)
        bn_var = np.ones(3, dtype=np.float32)

        initializers_list = [
            W,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Conv"
        assert len(result[0].input) == 3

    def test_fuse_conv_bn_multiple_channels(self):
        """Test Conv->BN fusion with multiple channels."""
        X = create_tensor_value_info("X", "float32", [1, 16, 64, 64])
        Y = create_tensor_value_info("Y", "float32", [1, 32, 64, 64])

        W = create_initializer("W", np.random.randn(32, 16, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(32, dtype=np.float32))
        bn_scale = np.ones(32, dtype=np.float32)
        bn_bias = np.zeros(32, dtype=np.float32)
        bn_mean = np.zeros(32, dtype=np.float32)
        bn_var = np.ones(32, dtype=np.float32)

        initializers_list = [
            W,
            B,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Conv"

    def test_fuse_conv_bn_with_nonzero_bn_bias(self):
        """Test Conv->BN fusion with non-zero BN bias."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([conv, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Conv"

    def test_fuse_conv_bn_no_fusion_when_branching(self):
        """Test no fusion when Conv output is used by other nodes."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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

        conv = helper.make_node("Conv", inputs=["X", "W", "B"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )
        relu = helper.make_node("Relu", inputs=["C"], outputs=["Z"])

        model = create_minimal_onnx_model([conv, bn, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 3  # No fusion due to branching

    def test_fuse_conv_bn_chain(self):
        """Test fusion of multiple Conv->BN patterns in sequence."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Z = create_tensor_value_info("Z", "float32", [1, 3, 32, 32])

        W1 = create_initializer("W1", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(3, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(3, dtype=np.float32))
        bn_scale1 = np.ones(3, dtype=np.float32)
        bn_bias1 = np.zeros(3, dtype=np.float32)
        bn_mean1 = np.zeros(3, dtype=np.float32)
        bn_var1 = np.ones(3, dtype=np.float32)
        bn_scale2 = np.ones(3, dtype=np.float32)
        bn_bias2 = np.zeros(3, dtype=np.float32)
        bn_mean2 = np.zeros(3, dtype=np.float32)
        bn_var2 = np.ones(3, dtype=np.float32)

        initializers_list = [
            W1,
            B1,
            W2,
            B2,
            create_initializer("bn_scale1", bn_scale1),
            create_initializer("bn_bias1", bn_bias1),
            create_initializer("bn_mean1", bn_mean1),
            create_initializer("bn_var1", bn_var1),
            create_initializer("bn_scale2", bn_scale2),
            create_initializer("bn_bias2", bn_bias2),
            create_initializer("bn_mean2", bn_mean2),
            create_initializer("bn_var2", bn_var2),
        ]

        conv1 = helper.make_node("Conv", inputs=["X", "W1", "B1"], outputs=["C1"])
        bn1 = helper.make_node(
            "BatchNormalization",
            inputs=["C1", "bn_scale1", "bn_bias1", "bn_mean1", "bn_var1"],
            outputs=["Y"],
        )
        conv2 = helper.make_node("Conv", inputs=["Y", "W2", "B2"], outputs=["C2"])
        bn2 = helper.make_node(
            "BatchNormalization",
            inputs=["C2", "bn_scale2", "bn_bias2", "bn_mean2", "bn_var2"],
            outputs=["Z"],
        )

        model = create_minimal_onnx_model([conv1, bn1, conv2, bn2], [X], [Z], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
        assert isinstance(result, list)
        assert len(result) == 2  # Two fused Conv nodes
        assert all(node.op_type == "Conv" for node in result)


class TestFuseBnConv:
    """Test BN+Conv fusion implementation."""

    def test_fuse_bn_conv_basic(self):
        """Test basic BN->Conv fusion."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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
            outputs=["B_out"],
        )
        conv = helper.make_node("Conv", inputs=["B_out", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "Conv"

    def test_fuse_bn_conv_with_padding(self):
        """Test BN->Conv fusion skipped when Conv has padding."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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
            outputs=["B_out"],
        )
        conv = helper.make_node(
            "Conv", inputs=["B_out", "W", "B"], outputs=["Y"], pads=[1, 1, 1, 1]
        )

        model = create_minimal_onnx_model([bn, conv], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
        assert isinstance(result, list)
        assert len(result) == 2  # No fusion due to padding

    def test_fuse_bn_conv_no_branching(self):
        """Test no fusion when BN has multiple consumers."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        W = create_initializer("W", np.random.randn(3, 3, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(3, dtype=np.float32))
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
            outputs=["B_out"],
        )
        conv = helper.make_node("Conv", inputs=["B_out", "W", "B"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["B_out"], outputs=["Z"])

        model = create_minimal_onnx_model([bn, conv, relu], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
        assert isinstance(result, list)
        assert len(result) == 3  # No fusion due to branching


class TestFuseConvTransposeBn:
    """Test ConvTranspose+BN fusion implementation."""

    def test_fuse_convtranspose_bn_basic(self):
        """Test basic ConvTranspose->BN fusion."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 64, 64])

        W = create_initializer("W", np.random.randn(3, 2, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(2, dtype=np.float32))
        bn_scale = np.ones(2, dtype=np.float32)
        bn_bias = np.zeros(2, dtype=np.float32)
        bn_mean = np.zeros(2, dtype=np.float32)
        bn_var = np.ones(2, dtype=np.float32)

        initializers_list = [
            W,
            B,
            create_initializer("bn_scale", bn_scale),
            create_initializer("bn_bias", bn_bias),
            create_initializer("bn_mean", bn_mean),
            create_initializer("bn_var", bn_var),
        ]

        convt = helper.make_node("ConvTranspose", inputs=["X", "W", "B"], outputs=["C"])
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["C", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
        )

        model = create_minimal_onnx_model([convt, bn], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_transpose_bn_or_bn_conv_transpose(
            nodes, initializers, is_conv_transpose_bn=True
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "ConvTranspose"

    def test_fuse_convtranspose_bn_no_fusion_non_pattern(self):
        """Test no fusion when nodes don't match pattern."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 3, 32, 32])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_transpose_bn_or_bn_conv_transpose(
            nodes, initializers, is_conv_transpose_bn=True
        )
        assert isinstance(result, list)
        assert len(result) == 1  # No fusion


class TestFuseBnConvTranspose:
    """Test BN+ConvTranspose fusion implementation."""

    def test_fuse_bn_convtranspose_basic(self):
        """Test basic BN->ConvTranspose fusion."""
        X = create_tensor_value_info("X", "float32", [1, 3, 32, 32])
        Y = create_tensor_value_info("Y", "float32", [1, 2, 64, 64])

        W = create_initializer("W", np.random.randn(3, 2, 3, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(2, dtype=np.float32))
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
            outputs=["B_out"],
        )
        convt = helper.make_node("ConvTranspose", inputs=["B_out", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([bn, convt], [X], [Y], initializers_list)
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_conv_transpose_bn_or_bn_conv_transpose(
            nodes, initializers, is_conv_transpose_bn=False
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].op_type == "ConvTranspose"
