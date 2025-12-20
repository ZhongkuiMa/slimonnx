"""ONNX model structure analysis utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "analyze_structure",
    "count_op_types",
    "analyze_inputs_outputs",
    "build_topology",
    "export_topology_json",
    "generate_text_report",
    "generate_json_report",
    "print_node_graph",
]

from slimonnx.slimonnx.analyzer import analyze_inputs_outputs, analyze_structure, count_op_types
from slimonnx.slimonnx.reporter import generate_json_report, generate_text_report, print_node_graph
from slimonnx.slimonnx.topology import build_topology, export_topology_json
