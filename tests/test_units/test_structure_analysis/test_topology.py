"""Tests for topology analysis and export."""

import json
import tempfile
from pathlib import Path

from onnx import TensorProto, helper

from slimonnx.structure_analysis.topology import build_topology, export_topology_json


def create_simple_model():
    """Create a simple ONNX model for topology testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")

    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


class TestBuildTopology:
    """Test build_topology function."""

    def test_build_topology_single_node(self):
        """Test building topology with single node."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert len(topology) == 1
        assert "relu_0" in topology
        assert topology["relu_0"]["op_type"] == "Relu"
        assert topology["relu_0"]["inputs"] == ["X"]
        assert topology["relu_0"]["outputs"] == ["Y"]
        assert "predecessors" in topology["relu_0"]
        assert "successors" in topology["relu_0"]

    def test_build_topology_multiple_nodes(self):
        """Test building topology with chained nodes."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert len(topology) == 2
        assert "relu_0" in topology
        assert "relu_1" in topology

        # Check predecessors/successors
        assert topology["relu_1"]["predecessors"] == ["relu_0"]
        assert topology["relu_0"]["successors"] == ["relu_1"]

    def test_build_topology_branching(self):
        """Test topology with branching paths."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        _Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
        W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")
        node3 = helper.make_node("Add", inputs=["Y", "Y"], outputs=["W"], name="add_0")

        graph = helper.make_graph([node1, node2, node3], "model", [X], [W])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        # relu_0 should have two successors (relu_1 and add_0)
        assert len(topology["relu_0"]["successors"]) == 2
        assert "relu_1" in topology["relu_0"]["successors"]
        assert "add_0" in topology["relu_0"]["successors"]

    def test_build_topology_unnamed_nodes(self):
        """Test topology building with unnamed nodes."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        # Create node without name
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        graph = helper.make_graph([node], "model", [X], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        # Should use op_type_unnamed as key
        assert "Relu_unnamed" in topology
        assert topology["Relu_unnamed"]["op_type"] == "Relu"

    def test_build_topology_no_input_nodes(self):
        """Test nodes with no predecessor (graph input)."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        # relu_0 should have no predecessors (X is graph input)
        assert topology["relu_0"]["predecessors"] == []

    def test_build_topology_no_output_nodes(self):
        """Test nodes with no successor (graph output)."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        # relu_0 should have no successors (Y is graph output)
        assert topology["relu_0"]["successors"] == []

    def test_build_topology_multiple_outputs(self):
        """Test node with multiple outputs."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        # Split operation (has multiple outputs)
        split_node = helper.make_node("Split", inputs=["X"], outputs=["Y", "Z"], name="split_0")

        graph = helper.make_graph([split_node], "model", [X], [Y, Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert topology["split_0"]["outputs"] == ["Y", "Z"]

    def test_build_topology_multiple_inputs(self):
        """Test node with multiple inputs."""
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        add_node = helper.make_node("Add", inputs=["X1", "X2"], outputs=["Y"], name="add_0")

        graph = helper.make_graph([add_node], "model", [X1, X2], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert topology["add_0"]["inputs"] == ["X1", "X2"]

    def test_build_topology_complex_graph(self):
        """Test with complex multi-layer graph."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 224, 224])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 224, 224])
        _Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 16, 224, 224])
        _W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 16, 224, 224])
        out = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 16, 224, 224])

        # Conv -> BN -> ReLU -> Add pattern
        conv = helper.make_node("Conv", inputs=["X", "W_conv"], outputs=["Y"], name="conv_0")
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["Y", "bn_scale", "bn_b", "bn_mean", "bn_var"],
            outputs=["Z"],
            name="bn_0",
        )
        relu = helper.make_node("ReLU", inputs=["Z"], outputs=["Z2"], name="relu_0")
        add = helper.make_node("Add", inputs=["Z2", "X"], outputs=["O"], name="add_0")

        graph = helper.make_graph(
            [conv, bn, relu, add],
            "model",
            [X],
            [out],
        )
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert len(topology) == 4
        # Check chain
        assert topology["conv_0"]["successors"] == ["bn_0"]
        assert topology["bn_0"]["successors"] == ["relu_0"]
        assert topology["relu_0"]["successors"] == ["add_0"]


class TestExportTopologyJson:
    """Test export_topology_json function."""

    def test_export_topology_basic(self):
        """Test exporting basic topology to JSON."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            assert output_path.exists()

            with output_path.open() as f:
                data = json.load(f)

            assert "nodes" in data
            assert "edges" in data
            assert "node_count" in data
            assert "edge_count" in data
            assert data["node_count"] == 1

    def test_export_topology_with_shapes(self):
        """Test exporting topology with shape information."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        data_shapes = {"X": [1, 3], "Y": [1, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path), data_shapes=data_shapes)

            with output_path.open() as f:
                data = json.load(f)

            # Check that shapes are included
            assert "shapes" in data["nodes"][0]
            assert data["nodes"][0]["shapes"]["Y"] == [1, 3]

    def test_export_topology_multiple_nodes(self):
        """Test exporting topology with multiple nodes."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            with output_path.open() as f:
                data = json.load(f)

            assert data["node_count"] == 2
            assert data["edge_count"] == 1  # One edge between relu_0 and relu_1

    def test_export_topology_nodes_structure(self):
        """Test that exported nodes have correct structure."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            with output_path.open() as f:
                data = json.load(f)

            node_info = data["nodes"][0]
            assert "name" in node_info
            assert "op_type" in node_info
            assert "inputs" in node_info
            assert "outputs" in node_info
            assert node_info["op_type"] == "Relu"
            assert node_info["inputs"] == ["X"]
            assert node_info["outputs"] == ["Y"]

    def test_export_topology_edges_structure(self):
        """Test that exported edges have correct structure."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            with output_path.open() as f:
                data = json.load(f)

            edge = data["edges"][0]
            assert "from" in edge
            assert "to" in edge
            assert "tensor" in edge
            assert edge["tensor"] == "Y"

    def test_export_topology_empty_shapes(self):
        """Test exporting with empty data_shapes dict."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path), data_shapes={})

            with output_path.open() as f:
                data = json.load(f)

            # Should not have shapes key if dict is empty
            assert "shapes" not in data["nodes"][0] or not data["nodes"][0].get("shapes")

    def test_export_topology_partial_shapes(self):
        """Test exporting with partial shape information."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        # Only provide shape for one output
        data_shapes = {"Y": [1, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path), data_shapes=data_shapes)

            with output_path.open() as f:
                data = json.load(f)

            # Should have shape for Y
            assert data["nodes"][0]["shapes"]["Y"] == [1, 3]

    def test_export_topology_json_valid(self):
        """Test that exported JSON is valid and readable."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            # Should be able to read and parse
            with output_path.open() as f:
                data = json.load(f)

            # Verify structure
            assert isinstance(data, dict)
            assert isinstance(data["nodes"], list)
            assert isinstance(data["edges"], list)
            assert isinstance(data["node_count"], int)
            assert isinstance(data["edge_count"], int)

    def test_export_topology_unnamed_node_name(self):
        """Test that unnamed nodes get proper names in export."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        # Create node without name
        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        graph = helper.make_graph([node], "model", [X], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            with output_path.open() as f:
                data = json.load(f)

            # Should use op_type_unnamed pattern
            assert data["nodes"][0]["name"] == "Relu_unnamed"

    def test_export_topology_branching_edges(self):
        """Test edge export with branching topology."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        _Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
        W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")
        node3 = helper.make_node("Add", inputs=["Y", "Y"], outputs=["W"], name="add_0")

        graph = helper.make_graph([node1, node2, node3], "model", [X], [W])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            export_topology_json(nodes, str(output_path))

            with output_path.open() as f:
                data = json.load(f)

            # Should have 3 edges: relu_0->relu_1, relu_0->add_0 (first Y), relu_0->add_0 (second Y)
            assert data["edge_count"] == 3
