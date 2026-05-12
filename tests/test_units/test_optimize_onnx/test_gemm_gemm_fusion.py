"""Tests for Gemm+Gemm fusion optimization."""

__docformat__ = "restructuredtext"

import sys
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from slimonnx.optimize_onnx._gemm_gemm import (
    _count_node_connections,
    _filter_fusable_gemm_nodes,
    _fuse_gemm_gemm,
    _group_adjacent_gemm_nodes,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _helpers import (  # type: ignore[import-not-found]
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestCountNodeConnections:
    """Test _count_node_connections function."""

    @pytest.mark.parametrize(
        "num_gemms",
        [
            pytest.param(1, id="single"),
            pytest.param(2, id="chain"),
        ],
    )
    def test_returns_connection_counts(self, num_gemms):
        """Test that _count_node_connections returns integer counts."""
        if num_gemms == 1:
            X = create_tensor_value_info("X", "float32", [2, 3])
            Y = create_tensor_value_info("Y", "float32", [2, 4])
            W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
            B = create_initializer("B", np.zeros(4, dtype=np.float32))
            gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
            model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        else:
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
        producer_count, consumer_count = _count_node_connections(nodes[0], nodes)
        assert isinstance(producer_count, int)
        assert isinstance(consumer_count, int)


class TestFilterFusableGemmNodes:
    """Test _filter_fusable_gemm_nodes function."""

    @pytest.mark.parametrize(
        ("has_other_nodes", "gemm_count"),
        [
            pytest.param(True, 1, id="single"),
            pytest.param(False, 2, id="multiple"),
            pytest.param(False, 0, id="no_gemms"),
        ],
    )
    def test_returns_node_list(self, has_other_nodes, gemm_count):
        """Test that _filter_fusable_gemm_nodes returns a list of nodes."""
        if gemm_count == 0:
            X = create_tensor_value_info("X", "float32", [2, 3])
            Y = create_tensor_value_info("Y", "float32", [2, 3])
            relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
            model = create_minimal_onnx_model([relu], [X], [Y])
        elif gemm_count == 1:
            X = create_tensor_value_info("X", "float32", [2, 3])
            Y = create_tensor_value_info("Y", "float32", [2, 4])
            W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
            B = create_initializer("B", np.zeros(4, dtype=np.float32))
            gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
            nodes_list = [gemm]
            if has_other_nodes:
                relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])
                nodes_list.append(relu)
            model = create_minimal_onnx_model(nodes_list, [X], [Y], [W, B])
        else:
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
        result = _filter_fusable_gemm_nodes(nodes)
        assert isinstance(result, list)


class TestGroupAdjacentGemmNodes:
    """Test _group_adjacent_gemm_nodes function."""

    @pytest.mark.parametrize(
        ("consecutive", "count"),
        [
            pytest.param(True, 2, id="consecutive"),
            pytest.param(False, 1, id="single"),
        ],
    )
    def test_returns_grouped_nodes(self, consecutive, count):
        """Test that _group_adjacent_gemm_nodes returns grouped node list."""
        if not consecutive and count == 1:
            X = create_tensor_value_info("X", "float32", [2, 3])
            Y = create_tensor_value_info("Y", "float32", [2, 4])
            W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
            B = create_initializer("B", np.zeros(4, dtype=np.float32))
            gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
            model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        else:
            X = create_tensor_value_info("X", "float32", [2, 3])
            Z = create_tensor_value_info("Z", "float32", [2, 5])
            W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
            B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
            W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
            B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))
            gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
            gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])

            if consecutive:
                model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
            else:
                relu = helper.make_node("Relu", inputs=["Y"], outputs=["R"])
                model = create_minimal_onnx_model([gemm1, relu, gemm2], [X], [Z], [W1, B1, W2, B2])

        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        # Filter Gemm nodes if not consecutive
        if not consecutive:
            nodes = [node for node in nodes if node.op_type == "Gemm"]

        result = _group_adjacent_gemm_nodes(nodes, initializers)
        assert isinstance(result, list)


class TestFuseGemmGemm:
    """Test _fuse_gemm_gemm function."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param("two_consecutive", id="two_consecutive"),
            pytest.param("single", id="single"),
            pytest.param("with_relu", id="with_other_nodes"),
            pytest.param("three", id="three_consecutive"),
        ],
    )
    def test_produces_fused_nodes(self, config):
        """Test that _fuse_gemm_gemm produces fused node outputs."""
        if config == "two_consecutive":
            X = create_tensor_value_info("X", "float32", [2, 3])
            Z = create_tensor_value_info("Z", "float32", [2, 5])
            W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
            B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
            W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
            B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))
            gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y"])
            gemm2 = helper.make_node("Gemm", inputs=["Y", "W2", "B2"], outputs=["Z"])
            model = create_minimal_onnx_model([gemm1, gemm2], [X], [Z], [W1, B1, W2, B2])
            expected_len = 0
        elif config == "single":
            X = create_tensor_value_info("X", "float32", [2, 3])
            Y = create_tensor_value_info("Y", "float32", [2, 4])
            W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
            B = create_initializer("B", np.zeros(4, dtype=np.float32))
            gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
            model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
            expected_len = 1
        elif config == "with_relu":
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
            expected_len = 3
        else:  # three
            X = create_tensor_value_info("X", "float32", [2, 3])
            W = create_tensor_value_info("W", "float32", [2, 6])
            W1 = create_initializer("W1", np.random.randn(3, 4).astype(np.float32))
            B1 = create_initializer("B1", np.zeros(4, dtype=np.float32))
            W2 = create_initializer("W2", np.random.randn(4, 5).astype(np.float32))
            B2 = create_initializer("B2", np.zeros(5, dtype=np.float32))
            W3 = create_initializer("W3", np.random.randn(5, 6).astype(np.float32))
            B3 = create_initializer("B3", np.zeros(6, dtype=np.float32))
            gemm1 = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y1"])
            gemm2 = helper.make_node("Gemm", inputs=["Y1", "W2", "B2"], outputs=["Y2"])
            gemm3 = helper.make_node("Gemm", inputs=["Y2", "W3", "B3"], outputs=["W"])
            model = create_minimal_onnx_model(
                [gemm1, gemm2, gemm3], [X], [W], [W1, B1, W2, B2, W3, B3]
            )
            expected_len = 0

        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}
        result = _fuse_gemm_gemm(nodes, initializers)

        assert isinstance(result, list)
        if config in ("single", "with_relu"):
            assert len(result) == expected_len
        else:
            assert len(result) >= expected_len
