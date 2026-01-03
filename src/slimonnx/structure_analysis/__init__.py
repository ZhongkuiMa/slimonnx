"""ONNX model structure analysis utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "analyze_inputs_outputs",
    "analyze_structure",
    "build_topology",
    "count_op_types",
    "export_topology_json",
    "generate_json_report",
    "generate_text_report",
    "print_node_graph",
]

from slimonnx.structure_analysis import analyzer as _analyzer_module
from slimonnx.structure_analysis import reporter as _reporter_module
from slimonnx.structure_analysis import topology as _topology_module

analyze_inputs_outputs = _analyzer_module.analyze_inputs_outputs
analyze_structure = _analyzer_module.analyze_structure
count_op_types = _analyzer_module.count_op_types

generate_json_report = _reporter_module.generate_json_report
generate_text_report = _reporter_module.generate_text_report
print_node_graph = _reporter_module.print_node_graph

build_topology = _topology_module.build_topology
export_topology_json = _topology_module.export_topology_json
