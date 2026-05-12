"""Tests for topology analysis and export."""

__docformat__ = "restructuredtext"

import json
from pathlib import Path

import pytest
from onnx import TensorProto, helper

from slimonnx.structure_analysis.topology import build_topology, export_topology_json


def create_simple_model():
    """Create a simple single-Relu ONNX model for topology testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")

    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


class TestBuildTopology:
    """Test build_topology function."""

    def test_single_node_produces_one_entry_with_metadata(self):
        """Test that a single-node graph produces one topology entry with all metadata keys."""
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

    def test_chained_nodes_link_predecessors_and_successors(self):
        """Test that successor/predecessor edges are set for a linear chain."""
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
        assert topology["relu_1"]["predecessors"] == ["relu_0"]
        assert topology["relu_0"]["successors"] == ["relu_1"]

    def test_branching_node_has_two_successors(self):
        """Test that a node whose output feeds two nodes has both in its successors list."""
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

        assert len(topology["relu_0"]["successors"]) == 2
        assert "relu_1" in topology["relu_0"]["successors"]
        assert "add_0" in topology["relu_0"]["successors"]

    def test_unnamed_node_keyed_by_op_type_suffix(self):
        """Test that a node without a name is stored under the '<OpType>_unnamed' key."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        graph = helper.make_graph([node], "model", [X], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert "Relu_unnamed" in topology
        assert topology["Relu_unnamed"]["op_type"] == "Relu"

    @pytest.mark.parametrize(
        ("direction", "key"),
        [("predecessors", "relu_0"), ("successors", "relu_0")],
        ids=["no_predecessors_at_graph_input", "no_successors_at_graph_output"],
    )
    def test_boundary_node_has_empty_neighbor_list(self, direction, key):
        """Test that graph-boundary nodes report empty predecessors/successors."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert topology[key][direction] == []

    def test_split_node_reports_multiple_outputs(self):
        """Test that a node with two outputs lists both in its outputs entry."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        split_node = helper.make_node("Split", inputs=["X"], outputs=["Y", "Z"], name="split_0")

        graph = helper.make_graph([split_node], "model", [X], [Y, Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert topology["split_0"]["outputs"] == ["Y", "Z"]

    def test_add_node_reports_multiple_inputs(self):
        """Test that a node with two inputs lists both in its inputs entry."""
        X1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 3])
        X2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        add_node = helper.make_node("Add", inputs=["X1", "X2"], outputs=["Y"], name="add_0")

        graph = helper.make_graph([add_node], "model", [X1, X2], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert topology["add_0"]["inputs"] == ["X1", "X2"]

    def test_conv_bn_relu_add_chain_has_correct_successor_links(self):
        """Test that a Conv-BN-ReLU-Add chain builds successor links correctly."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 224, 224])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 224, 224])
        _Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 16, 224, 224])
        _W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 16, 224, 224])
        out = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 16, 224, 224])

        conv = helper.make_node("Conv", inputs=["X", "W_conv"], outputs=["Y"], name="conv_0")
        bn = helper.make_node(
            "BatchNormalization",
            inputs=["Y", "bn_scale", "bn_b", "bn_mean", "bn_var"],
            outputs=["Z"],
            name="bn_0",
        )
        relu = helper.make_node("ReLU", inputs=["Z"], outputs=["Z2"], name="relu_0")
        add = helper.make_node("Add", inputs=["Z2", "X"], outputs=["O"], name="add_0")

        graph = helper.make_graph([conv, bn, relu, add], "model", [X], [out])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        topology = build_topology(nodes)

        assert len(topology) == 4
        assert topology["conv_0"]["successors"] == ["bn_0"]
        assert topology["bn_0"]["successors"] == ["relu_0"]
        assert topology["relu_0"]["successors"] == ["add_0"]


class TestExportTopologyJson:
    """Test export_topology_json function."""

    def test_output_file_contains_required_keys(self, tmp_path: Path):
        """Test that the exported JSON has nodes, edges, node_count, and edge_count keys."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        assert output_path.exists()

        with output_path.open() as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert "node_count" in data
        assert "edge_count" in data
        assert data["node_count"] == 1

    def test_output_includes_shapes_when_data_shapes_provided(self, tmp_path: Path):
        """Test that shape info is embedded per-node when data_shapes is passed."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        data_shapes = {"X": [1, 3], "Y": [1, 3]}

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path), data_shapes=data_shapes)

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["nodes"]) > 0
        assert "shapes" in data["nodes"][0]
        assert data["nodes"][0]["shapes"]["Y"] == [1, 3]

    def test_counts_match_actual_nodes_and_edges(self, tmp_path: Path):
        """Test that node_count and edge_count match the graph contents."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        assert data["node_count"] == 2
        assert data["edge_count"] == 1

    def test_each_node_entry_has_name_op_type_inputs_outputs(self, tmp_path: Path):
        """Test that every node dict in the JSON has the four required fields."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["nodes"]) > 0
        node_info = data["nodes"][0]
        assert "name" in node_info
        assert "op_type" in node_info
        assert "inputs" in node_info
        assert "outputs" in node_info
        assert node_info["op_type"] == "Relu"
        assert node_info["inputs"] == ["X"]
        assert node_info["outputs"] == ["Y"]

    def test_each_edge_entry_has_from_to_tensor(self, tmp_path: Path):
        """Test that every edge dict in the JSON has from, to, and tensor fields."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], name="relu_0")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], name="relu_1")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["edges"]) > 0
        edge = data["edges"][0]
        assert "from" in edge
        assert "to" in edge
        assert "tensor" in edge
        assert edge["tensor"] == "Y"

    def test_empty_data_shapes_produces_no_shapes_field(self, tmp_path: Path):
        """Test that passing an empty data_shapes dict omits shapes from node entries."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path), data_shapes={})

        with output_path.open() as f:
            data = json.load(f)

        assert "shapes" not in data["nodes"][0] or not data["nodes"][0].get("shapes")

    def test_partial_data_shapes_only_annotates_available_tensors(self, tmp_path: Path):
        """Test that only the tensors present in data_shapes get shape annotations."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        data_shapes = {"Y": [1, 3]}

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path), data_shapes=data_shapes)

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["nodes"]) > 0
        assert data["nodes"][0]["shapes"]["Y"] == [1, 3]

    def test_output_is_valid_json_with_typed_fields(self, tmp_path: Path):
        """Test that the exported file is parseable JSON with correctly typed top-level fields."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert isinstance(data["node_count"], int)
        assert isinstance(data["edge_count"], int)

    def test_unnamed_node_exported_with_op_type_unnamed_name(self, tmp_path: Path):
        """Test that an unnamed node is exported with '<OpType>_unnamed' as its name."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

        graph = helper.make_graph([node], "model", [X], [Y])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["nodes"]) > 0
        assert data["nodes"][0]["name"] == "Relu_unnamed"

    def test_branching_topology_exports_three_edges(self, tmp_path: Path):
        """Test that a fork topology (relu_0 feeds both relu_1 and add_0) exports 3 edges."""
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

        output_path = tmp_path / "topology.json"
        export_topology_json(nodes, str(output_path))

        with output_path.open() as f:
            data = json.load(f)

        # relu_0->relu_1, relu_0->add_0 (first Y), relu_0->add_0 (second Y)
        assert data["edge_count"] == 3
