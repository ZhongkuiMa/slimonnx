"""Tests for structure analysis report generation."""

import json
import tempfile
from pathlib import Path
from typing import Any

from onnx import TensorProto, helper

from slimonnx.structure_analysis.reporter import (
    generate_json_report,
    generate_text_report,
    print_node_graph,
)


def create_simple_model():
    """Create a simple ONNX model for testing."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])

    graph = helper.make_graph([node], "test_model", [X], [Y])
    model = helper.make_model(graph)
    return model


class TestPrintNodeGraph:
    """Test print_node_graph function."""

    def test_print_single_node(self):
        """Test printing single node graph."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output = print_node_graph(nodes)

        # Should contain node information
        assert "Relu" in output
        assert "X" in output
        assert "Y" in output
        assert "->" in output

    def test_print_multiple_nodes(self):
        """Test printing multiple nodes."""
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        _Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])

        node1 = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        node2 = helper.make_node("Relu", inputs=["Y"], outputs=["Z"])

        graph = helper.make_graph([node1, node2], "model", [X], [Z])
        model = helper.make_model(graph)
        nodes = list(model.graph.node)

        output = print_node_graph(nodes)

        # Should contain both nodes
        assert output.count("Relu") >= 2
        assert "X" in output
        assert "Z" in output

    def test_print_with_shapes(self):
        """Test printing with shape information."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        data_shapes = {"X": [1, 3], "Y": [1, 3]}

        output = print_node_graph(nodes, data_shapes=data_shapes)

        # Should contain shape information
        assert "1x3" in output or "1" in output

    def test_print_empty_node_list(self):
        """Test printing with empty node list."""
        output = print_node_graph([])

        # Should return empty or minimal string
        assert isinstance(output, str)

    def test_print_returns_string(self):
        """Test that function returns string."""
        model = create_simple_model()
        nodes = list(model.graph.node)

        output = print_node_graph(nodes)

        assert isinstance(output, str)


class TestGenerateTextReport:
    """Test generate_text_report function."""

    def test_generate_basic_report(self):
        """Test generating basic text report."""
        analysis = {
            "model_name": "test_model",
            "num_nodes": 1,
            "num_initializers": 0,
            "input_shapes": {"X": [1, 3]},
            "output_shapes": {"Y": [1, 3]},
        }

        report = generate_text_report(analysis)

        # Should contain header and analysis
        assert "ANALYSIS REPORT" in report
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_required_sections(self):
        """Test that report contains required sections."""
        analysis = {
            "model_name": "test_model",
            "num_nodes": 5,
            "num_initializers": 2,
            "input_shapes": {},
            "output_shapes": {},
        }

        report = generate_text_report(analysis)

        # Should be formatted text
        assert "=" * 10 in report or "=" in report
        assert isinstance(report, str)

    def test_report_with_complex_analysis(self):
        """Test report with complex analysis data."""
        analysis = {
            "model_name": "complex_model",
            "num_nodes": 25,
            "num_initializers": 10,
            "num_inputs": 1,
            "num_outputs": 1,
            "input_shapes": {"X": [1, 3, 224, 224]},
            "output_shapes": {"Y": [1, 1000]},
            "patterns_detected": ["conv_bn", "matmul_add"],
        }

        report = generate_text_report(analysis)

        # Should handle complex data
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_is_readable(self):
        """Test that report is human-readable."""
        analysis = {"model_name": "test", "num_nodes": 1}

        report = generate_text_report(analysis)

        # Should have multiple lines for readability
        lines = report.split("\n")
        assert len(lines) > 1

    def test_report_with_model_path(self):
        """Test report includes model path."""
        analysis = {
            "model_path": "/path/to/model.onnx",
            "model_name": "test_model",
        }

        report = generate_text_report(analysis)

        # Should contain model path
        assert "Model:" in report
        assert "/path/to/model.onnx" in report

    def test_report_with_structure(self):
        """Test report includes structure information."""
        analysis = {
            "structure": {
                "node_count": 10,
                "initializer_count": 5,
                "num_inputs": 1,
                "num_outputs": 1,
                "op_type_counts": {"Conv": 3, "Relu": 2, "Add": 1},
            }
        }

        report = generate_text_report(analysis)

        # Should contain structure section
        assert "STRUCTURE:" in report
        assert "Nodes: 10" in report
        assert "Initializers: 5" in report
        assert "Inputs: 1" in report
        assert "Outputs: 1" in report
        assert "Conv: 3" in report
        assert "Relu: 2" in report

    def test_report_with_structure_no_op_types(self):
        """Test report with structure but no op_type_counts."""
        analysis = {
            "structure": {
                "node_count": 5,
                "initializer_count": 2,
                "num_inputs": 1,
                "num_outputs": 1,
            }
        }

        report = generate_text_report(analysis)

        # Should include structure without op_type_counts
        assert "STRUCTURE:" in report
        assert "Nodes: 5" in report

    def test_report_with_validation(self):
        """Test report includes validation information."""
        analysis = {
            "validation": {
                "is_valid": True,
                "onnx_checker": {"valid": True},
                "runtime": {"can_load": True},
                "dead_nodes": [],
                "broken_connections": [],
                "orphan_initializers": [],
            }
        }

        report = generate_text_report(analysis)

        # Should contain validation section
        assert "VALIDATION:" in report
        assert "Valid: True" in report
        assert "ONNX Checker: True" in report
        assert "Runtime Load: True" in report
        assert "Dead Nodes: 0" in report

    def test_report_with_validation_issues(self):
        """Test report with validation issues."""
        analysis = {
            "validation": {
                "is_valid": False,
                "onnx_checker": {"valid": False},
                "runtime": {"can_load": False},
                "dead_nodes": ["node1", "node2"],
                "broken_connections": ["conn1"],
                "orphan_initializers": ["init1"],
            }
        }

        report = generate_text_report(analysis)

        # Should contain dead nodes info
        assert "VALIDATION:" in report
        assert "Valid: False" in report
        assert "Dead nodes:" in report
        assert "node1" in report

    def test_report_with_patterns(self):
        """Test report includes detected patterns."""
        analysis = {
            "patterns": {
                "conv_bn": {
                    "count": 5,
                    "description": "Conv+BatchNorm fusion",
                },
                "matmul_add": {
                    "count": 3,
                    "description": "MatMul+Add fusion",
                },
            }
        }

        report = generate_text_report(analysis)

        # Should contain patterns section
        assert "PATTERNS DETECTED:" in report
        assert "Conv+BatchNorm fusion: 5" in report
        assert "MatMul+Add fusion: 3" in report

    def test_report_with_optimization_opportunities(self):
        """Test report includes optimization opportunities."""
        analysis = {
            "optimization_opportunities": {
                "total_fusible": 8,
                "total_redundant": 2,
                "estimated_reduction": 10,
            }
        }

        report = generate_text_report(analysis)

        # Should contain optimization section
        assert "OPTIMIZATION OPPORTUNITIES:" in report
        assert "Fusible patterns: 8" in report
        assert "Redundant operations: 2" in report
        assert "Estimated node reduction: 10" in report

    def test_report_with_all_sections(self):
        """Test report with all possible sections."""
        analysis = {
            "model_path": "/path/to/model.onnx",
            "structure": {
                "node_count": 10,
                "initializer_count": 5,
                "num_inputs": 1,
                "num_outputs": 1,
                "op_type_counts": {"Conv": 3, "Relu": 2},
            },
            "validation": {
                "is_valid": True,
                "onnx_checker": {"valid": True},
                "runtime": {"can_load": True},
                "dead_nodes": [],
                "broken_connections": [],
                "orphan_initializers": [],
            },
            "patterns": {
                "conv_bn": {"count": 2, "description": "Conv+BN"},
            },
            "optimization_opportunities": {
                "total_fusible": 5,
                "total_redundant": 1,
                "estimated_reduction": 6,
            },
        }

        report = generate_text_report(analysis)

        # Should contain all sections
        assert "ONNX MODEL ANALYSIS REPORT" in report
        assert "Model:" in report
        assert "STRUCTURE:" in report
        assert "VALIDATION:" in report
        assert "PATTERNS DETECTED:" in report
        assert "OPTIMIZATION OPPORTUNITIES:" in report


class TestGenerateJsonReport:
    """Test generate_json_report function."""

    def test_generate_json_report(self):
        """Test generating JSON report file."""
        analysis = {
            "model_name": "test_model",
            "num_nodes": 1,
            "num_initializers": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(analysis, str(output_path))

            # File should be created
            assert output_path.exists()

            # Should be valid JSON
            with output_path.open() as f:
                parsed = json.load(f)
            assert parsed["model_name"] == "test_model"
            assert parsed["num_nodes"] == 1

    def test_json_preserves_data(self):
        """Test that JSON report preserves all data."""
        analysis = {
            "model_name": "complex_model",
            "num_nodes": 25,
            "num_initializers": 10,
            "input_shapes": {"X": [1, 3, 224, 224]},
            "patterns": ["conv_bn", "matmul_add"],
            "statistics": {"avg_node_size": 1024, "total_params": 1000000},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(analysis, str(output_path))

            with output_path.open() as f:
                parsed = json.load(f)

            # All data should be preserved
            assert parsed["model_name"] == analysis["model_name"]
            assert parsed["num_nodes"] == analysis["num_nodes"]
            assert parsed["input_shapes"] == analysis["input_shapes"]

    def test_json_with_nested_data(self):
        """Test JSON report with nested data structures."""
        analysis = {
            "model": {
                "name": "test",
                "layers": [
                    {"type": "Conv", "filters": 64},
                    {"type": "ReLU"},
                ],
            },
            "metrics": {"accuracy": 0.95, "latency": 12.5},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(analysis, str(output_path))

            with output_path.open() as f:
                parsed = json.load(f)

            # Should handle nested structures
            assert parsed["model"]["layers"][0]["filters"] == 64

    def test_json_creates_file(self):
        """Test that JSON report creates a file."""
        analysis = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            generate_json_report(analysis, str(output_path))

            # File should be created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_json_with_empty_analysis(self):
        """Test JSON report with empty analysis."""
        analysis: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generate_json_report(analysis, str(output_path))

            # Should create valid JSON file
            with output_path.open() as f:
                parsed = json.load(f)
            assert isinstance(parsed, dict)
