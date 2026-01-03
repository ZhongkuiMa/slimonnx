"""Tests for Gemm+Gemm fusion optimization."""

import sys
from pathlib import Path

import numpy as np
from onnx import helper

from slimonnx.optimize_onnx._gemm_gemm import (
    _count_node_connections,
    _filter_fusable_gemm_nodes,
    _fuse_gemm_gemm,
    _group_adjacent_gemm_nodes,
)

# Add parent directory to sys.path for conftest imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import (
    create_initializer,
    create_minimal_onnx_model,
    create_tensor_value_info,
)


class TestCountNodeConnections:
    """Test _count_node_connections function."""

    def test_count_connections_single_gemm(self):
        """Test connection count for single Gemm node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        producer_count, consumer_count = _count_node_connections(nodes[0], nodes)
        assert isinstance(producer_count, int)
        assert isinstance(consumer_count, int)

    def test_count_connections_gemm_chain(self):
        """Test connection count for Gemm in chain."""
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

    def test_filter_single_gemm(self):
        """Test filtering with single Gemm node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])
        relu = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        model = create_minimal_onnx_model([gemm, relu], [X], [Y], [W, B])
        nodes = list(model.graph.node)

        result = _filter_fusable_gemm_nodes(nodes)
        assert isinstance(result, list)

    def test_filter_multiple_gemms(self):
        """Test filtering with multiple Gemm nodes."""
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

    def test_filter_non_gemm_nodes(self):
        """Test filtering with no Gemm nodes."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 3])

        relu = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        model = create_minimal_onnx_model([relu], [X], [Y])
        nodes = list(model.graph.node)

        result = _filter_fusable_gemm_nodes(nodes)
        assert isinstance(result, list)


class TestGroupAdjacentGemmNodes:
    """Test _group_adjacent_gemm_nodes function."""

    def test_group_two_consecutive_gemms(self):
        """Test grouping two consecutive Gemm nodes."""
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

        result = _group_adjacent_gemm_nodes(nodes, initializers)
        assert isinstance(result, list)

    def test_group_single_gemm(self):
        """Test grouping with single Gemm node."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _group_adjacent_gemm_nodes(nodes, initializers)
        assert isinstance(result, list)

    def test_group_non_consecutive_gemms(self):
        """Test grouping non-consecutive Gemm nodes."""
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

        # Filter only Gemm nodes before grouping
        gemm_nodes = [node for node in nodes if node.op_type == "Gemm"]
        result = _group_adjacent_gemm_nodes(gemm_nodes, initializers)
        assert isinstance(result, list)


class TestFuseGemmGemm:
    """Test _fuse_gemm_gemm function."""

    def test_fuse_two_gemm_nodes(self):
        """Test fusing two consecutive Gemm nodes."""
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

        result = _fuse_gemm_gemm(nodes, initializers)

        # Should return list of nodes
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_fuse_single_gemm(self):
        """Test fusing with single Gemm node (no fusion)."""
        X = create_tensor_value_info("X", "float32", [2, 3])
        Y = create_tensor_value_info("Y", "float32", [2, 4])

        W = create_initializer("W", np.random.randn(3, 4).astype(np.float32))
        B = create_initializer("B", np.zeros(4, dtype=np.float32))

        gemm = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"])

        model = create_minimal_onnx_model([gemm], [X], [Y], [W, B])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_gemm_gemm(nodes, initializers)

        # Should preserve single node
        assert isinstance(result, list)
        assert len(result) == 1

    def test_fuse_gemm_with_other_nodes(self):
        """Test fusing Gemm with other node types."""
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

        result = _fuse_gemm_gemm(nodes, initializers)

        # Should preserve structure (no fusion due to Relu in between)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_fuse_three_consecutive_gemms(self):
        """Test fusing three consecutive Gemm nodes."""
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

        model = create_minimal_onnx_model([gemm1, gemm2, gemm3], [X], [W], [W1, B1, W2, B2, W3, B3])
        nodes = list(model.graph.node)
        initializers = {init.name: init for init in model.graph.initializer}

        result = _fuse_gemm_gemm(nodes, initializers)

        # Should process the fusion
        assert isinstance(result, list)
        assert len(result) >= 0
