"""Tests for Gemm chain pattern detection."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.pattern_detect.gemm_chains import (
    _get_gemm_attributes,
    detect_gemm_gemm,
)
from slimonnx.pattern_detect.utils import is_consecutive_nodes as _is_consecutive_nodes

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestGetGemmAttributes:
    """Test _get_gemm_attributes function."""

    @pytest.mark.parametrize(
        ("alpha", "beta", "trans_a", "trans_b"),
        [
            (1.0, 1.0, 0, 0),  # defaults
            (2.0, 0.5, 0, 0),  # custom alpha/beta
            (1.0, 1.0, 0, 1),  # with trans_b
            (1.5, 0.75, 1, 1),  # all set
        ],
    )
    def test_extracts_attributes(self, alpha, beta, trans_a, trans_b):
        """Test extraction of Gemm attributes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        kwargs = {}
        if alpha != 1.0:
            kwargs["alpha"] = alpha
        if beta != 1.0:
            kwargs["beta"] = beta
        if trans_a != 0:
            kwargs["transA"] = trans_a
        if trans_b != 0:
            kwargs["transB"] = trans_b

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], **kwargs)

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        attrs = _get_gemm_attributes(nodes[0])
        assert attrs["alpha"] == alpha
        assert attrs["beta"] == beta
        assert attrs["transA"] == trans_a
        assert attrs["transB"] == trans_b


class TestIsConsecutiveNodesGemm:
    """Test _is_consecutive_nodes function for Gemm patterns."""

    def test_detects_consecutive_gemm_pair(self):
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

    def test_fails_with_non_matching_outputs(self):
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

    def test_fails_with_multiple_consumers(self):
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

    def test_detects_fusible_gemm_pair(self):
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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"transA": 1},  # transA makes non-fusible
            {"alpha": 2.0},  # custom alpha makes non-fusible
        ],
    )
    def test_detects_non_fusible_patterns(self, kwargs):
        """Test Gemm+Gemm with non-fusible attributes."""
        X = create_tensor_value_info("X", "float32", [2, 3] if "transA" not in kwargs else [3, 2])
        Z = create_tensor_value_info("Z", "float32", [2, 5])

        W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
        B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
        W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
        B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))

        gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"], **kwargs)
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

    def test_misses_non_consecutive_gemm_pair(self):
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

    def test_misses_gemm_pair_with_missing_weights(self):
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

    def test_detects_multiple_gemm_chains(self):
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
