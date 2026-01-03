"""Tests for ONNX model cleanup utilities."""

from onnx import TensorProto, helper

from slimonnx.preprocess.cleanup import cleanup_model, clear_docstrings, mark_slimonnx_model


def create_simple_model_with_docstrings():
    """Create a simple ONNX model with docstrings."""
    # Create input and output
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    # Create a simple node with docstring
    node = helper.make_node(
        "Relu",
        inputs=["X"],
        outputs=["Y"],
        doc_string="This is a ReLU activation",
    )

    # Create graph
    graph = helper.make_graph(
        [node],
        "test_model",
        [X],
        [Y],
    )

    # Create model
    model = helper.make_model(graph)
    model.doc_string = "Original model with docstring"
    return model


class TestClearDocstrings:
    """Test clear_docstrings function."""

    def test_clear_node_docstrings(self):
        """Test clearing docstrings from nodes."""
        model = create_simple_model_with_docstrings()

        # Verify docstring exists before
        assert model.graph.node[0].doc_string == "This is a ReLU activation"

        # Clear docstrings
        cleaned = clear_docstrings(model)

        # Verify docstring cleared
        assert cleaned.graph.node[0].doc_string == ""

    def test_clear_docstrings_multiple_nodes(self):
        """Test clearing docstrings from multiple nodes."""
        # Create model with multiple nodes
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"], doc_string="Node 1")
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"], doc_string="Node 2")

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)

        # Clear docstrings
        cleaned = clear_docstrings(model)

        # All docstrings should be cleared
        for node in cleaned.graph.node:
            assert node.doc_string == ""

    def test_clear_docstrings_preserves_structure(self):
        """Test that clearing docstrings preserves model structure."""
        model = create_simple_model_with_docstrings()
        original_nodes_count = len(model.graph.node)

        cleaned = clear_docstrings(model)

        # Structure should be preserved
        assert len(cleaned.graph.node) == original_nodes_count
        assert cleaned.graph.node[0].op_type == model.graph.node[0].op_type


class TestMarkSlimONNXModel:
    """Test mark_slimonnx_model function."""

    def test_mark_producer_name(self):
        """Test marking producer name."""
        model = create_simple_model_with_docstrings()

        # Mark as SlimONNX
        marked = mark_slimonnx_model(model)

        # Check producer name
        assert "SlimONNX" in marked.producer_name
        assert "1.0.0" in marked.producer_name

    def test_mark_with_custom_version(self):
        """Test marking with custom version."""
        model = create_simple_model_with_docstrings()

        # Mark with custom version
        marked = mark_slimonnx_model(model, version="2.0.0")

        # Check custom version in producer name
        assert "2.0.0" in marked.producer_name

    def test_mark_model_docstring(self):
        """Test that model docstring is set."""
        model = create_simple_model_with_docstrings()

        marked = mark_slimonnx_model(model)

        # Check model docstring
        assert "SlimONNX" in marked.doc_string
        assert "1.0.0" in marked.doc_string

    def test_mark_with_different_versions(self):
        """Test marking with different version strings."""
        model = create_simple_model_with_docstrings()

        versions = ["1.0.0", "1.5.0", "2.1.0", "0.9.0"]
        for version in versions:
            marked = mark_slimonnx_model(model, version=version)
            assert version in marked.producer_name
            assert version in marked.doc_string


class TestCleanupModel:
    """Test cleanup_model function."""

    def test_cleanup_with_all_options_enabled(self):
        """Test cleanup with all options enabled."""
        model = create_simple_model_with_docstrings()

        cleaned = cleanup_model(model, clear_docs=True, mark_producer=True)

        # Both operations should be applied
        assert cleaned.graph.node[0].doc_string == ""
        assert "SlimONNX" in cleaned.producer_name

    def test_cleanup_clear_docs_only(self):
        """Test cleanup with only clear_docs enabled."""
        model = create_simple_model_with_docstrings()
        original_producer = model.producer_name

        cleaned = cleanup_model(model, clear_docs=True, mark_producer=False)

        # Only docs should be cleared, producer unchanged
        assert cleaned.graph.node[0].doc_string == ""
        assert cleaned.producer_name == original_producer

    def test_cleanup_mark_producer_only(self):
        """Test cleanup with only mark_producer enabled."""
        model = create_simple_model_with_docstrings()
        original_doc = model.graph.node[0].doc_string

        cleaned = cleanup_model(model, clear_docs=False, mark_producer=True)

        # Only producer should be marked, docs preserved
        assert cleaned.graph.node[0].doc_string == original_doc
        assert "SlimONNX" in cleaned.producer_name

    def test_cleanup_no_operations(self):
        """Test cleanup with all options disabled."""
        model = create_simple_model_with_docstrings()
        original_producer = model.producer_name
        original_doc = model.graph.node[0].doc_string

        cleaned = cleanup_model(model, clear_docs=False, mark_producer=False)

        # Nothing should change
        assert cleaned.graph.node[0].doc_string == original_doc
        assert cleaned.producer_name == original_producer

    def test_cleanup_with_custom_version(self):
        """Test cleanup with custom SlimONNX version."""
        model = create_simple_model_with_docstrings()

        cleaned = cleanup_model(
            model, clear_docs=True, mark_producer=True, slimonnx_version="2.5.0"
        )

        # Custom version should be used
        assert "2.5.0" in cleaned.producer_name
        assert "2.5.0" in cleaned.doc_string

    def test_cleanup_returns_model(self):
        """Test that cleanup returns a model object."""
        model = create_simple_model_with_docstrings()

        cleaned = cleanup_model(model)

        # Should return a valid model
        assert hasattr(cleaned, "graph")
        assert hasattr(cleaned, "producer_name")
        assert hasattr(cleaned, "doc_string")
