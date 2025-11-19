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

from .analyzer import analyze_structure, count_op_types, analyze_inputs_outputs
from .topology import build_topology, export_topology_json
from .reporter import generate_text_report, generate_json_report, print_node_graph
