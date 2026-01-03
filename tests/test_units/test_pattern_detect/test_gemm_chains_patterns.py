"""Tests for Gemm chain pattern detection."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.pattern_detect.gemm_chains import (
    _get_gemm_attributes,
    detect_gemm_gemm,
)
from slimonnx.pattern_detect.utils import is_consecutive_nodes as _is_consecutive_nodes

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGetGemmAttributes:
    """Test _get_gemm_attributes function."""

    def test_get_gemm_attributes_defaults(self):
        """Test extraction of default Gemm attributes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        attrs = _get_gemm_attributes(nodes[0])
        assert attrs["alpha"] == 1.0
        assert attrs["beta"] == 1.0
        assert attrs["transA"] == 0
        assert attrs["transB"] == 0

    def test_get_gemm_attributes_custom_alpha_beta(self):
        """Test extraction of custom alpha and beta values."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], alpha=2.0, beta=0.5)

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        attrs = _get_gemm_attributes(nodes[0])
        assert attrs["alpha"] == 2.0
        assert attrs["beta"] == 0.5
        assert attrs["transA"] == 0
        assert attrs["transB"] == 0

    def test_get_gemm_attributes_with_transposition(self):
        """Test extraction with transA and transB."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(4, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], transB=1)

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        attrs = _get_gemm_attributes(nodes[0])
        assert attrs["transA"] == 0
        assert attrs["transB"] == 1

    def test_get_gemm_attributes_all_attributes(self):
        """Test extraction with all attributes set."""
        X = create_tensor_value_info("X", "float32", [3, 2])
        Y = create_tensor_value_info("Y", "float32", [4, 2])

        W = create_initializer("W", np.random.randn(4, 3).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node(
            "Gemm", inputs=["X", "W", "B"], outputs=["Y"], alpha=1.5, beta=0.75, transA=1, transB=1
        )

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        attrs = _get_gemm_attributes(nodes[0])
        assert attrs["alpha"] == 1.5
        assert attrs["beta"] == 0.75
        assert attrs["transA"] == 1
        assert attrs["transB"] == 1


class TestIsConsecutiveNodesGemm:
    """Test _is_consecutive_nodes function for Gemm patterns."""

    def test_is_consecutive_gemm_gemm(self):
        """Test consecutive Gemm->Gemm detection."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is True

    def test_is_consecutive_gemm_false_different_outputs(self):
        """Test non-consecutive when outputs don't match."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(5, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Z", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Y], [W1, B1, W2, B2])
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is False

    def test_is_consecutive_gemm_multiple_consumers(self):
        """Test non-consecutive when Gemm output has multiple consumers."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])
        identity = helper.make_node("Identity", inputs=["Y"], outputs=["W"])

        model = create_minimal_onnx_model([gemm1, gemm2, identity], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)

        is_consecutive = _is_consecutive_nodes(nodes[0], nodes[1], nodes)
        assert is_consecutive is False


class TestDetectGemmGemm:
    """Test detect_gemm_gemm function."""

    def test_detect_gemm_gemm_fusible(self):
        """Test detection of fusible Gemm+Gemm pattern."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["gemm1_node"] == gemm1.name
        assert result[0]["gemm2_node"] == gemm2.name
        assert result[0]["can_fuse"] is True

    def test_detect_gemm_gemm_non_fusible_trans_a(self):
        """Test Gemm+Gemm with transA=1 (non-fusible)."""
        X = create_tensor_value_info("X", "float32", [3, 2])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"], transA=1)
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["can_fuse"] is False

    def test_detect_gemm_gemm_non_fusible_alpha(self):
        """Test Gemm+Gemm with alpha != 1 (non-fusible)."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"], alpha=2.0)
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["can_fuse"] is False

    def test_detect_gemm_gemm_no_pattern(self):
        """Test no detection when pattern missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_gemm_gemm_non_consecutive(self):
        """Test no detection when Gemm nodes not consecutive."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["R"])
        gemm2 = helper.make_node("Gemm", inputs=["R", "W2", "B2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, relu, gemm2], [X], [Z], [W1, B1, W2, B2])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_gemm_gemm_missing_weights(self):
        """Test no detection when weights missing."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "missing_w2", "missing_b2"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_gemm_gemm_multiple_chains(self):
        """Test detection of multiple Gemm+Gemm chains."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))
        W3 = create_initializer("W3", np.random.randn(5, 6).astype(np.float32))
        B3 = create_initializer("B3", np.zeros(6, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
        gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Y2"])
        gemm3 = helper.make_node("Gemm", inputs=["Y2", "W3", "B3"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm1, gemm2, gemm3], [X], [Z], [W1, B1, W2, B2, W3, B3])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = detect_gemm_gemm(nodes, initializers)
        assert isinstance(result, list)
        assert len(result) == 2  # Two patterns: (gemm1, gemm2) and (gemm2, gemm3)
        assert all(r["can_fuse"] is True for r in result)
