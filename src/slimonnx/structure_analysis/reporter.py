"""Report generation utilities."""

__docformat__ = "restructuredtext"
__all__ = ["generate_json_report", "generate_text_report", "print_node_graph"]

import json
from pathlib import Path

from onnx import NodeProto


def print_node_graph(
    nodes: list[NodeProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> str:
    """Format node graph as text.

    :param nodes: Model nodes
    :param data_shapes: Inferred shapes
    :return: Formatted node graph string
    """
    lines = []
    for i, node in enumerate(nodes, 1):
        node_name = node.name or f"{node.op_type}_{i}"

        # Format inputs
        inputs_str = []
        for inp in node.input:
            if data_shapes and inp in data_shapes:
                shape_str = "x".join(map(str, data_shapes[inp]))
                inputs_str.append(f"{inp}({shape_str})")
            else:
                inputs_str.append(inp)

        # Format outputs
        outputs_str = []
        for out in node.output:
            if data_shapes and out in data_shapes:
                shape_str = "x".join(map(str, data_shapes[out]))
                outputs_str.append(f"{out}({shape_str})")
            else:
                outputs_str.append(out)

        line = f"{node_name} [{node.op_type}]: {' + '.join(inputs_str)} -> {', '.join(outputs_str)}"
        lines.append(line)

    return "\n".join(lines)


def generate_text_report(analysis: dict) -> str:
    """Generate human-readable text report.

    :param analysis: Analysis results dictionary
    :return: Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ONNX MODEL ANALYSIS REPORT")
    lines.append("=" * 80)

    # Model path
    if "model_path" in analysis:
        lines.append(f"\nModel: {analysis['model_path']}")

    # Structure
    if "structure" in analysis:
        struct = analysis["structure"]
        lines.append("\nSTRUCTURE:")
        lines.append(f"  Nodes: {struct['node_count']}")
        lines.append(f"  Initializers: {struct['initializer_count']}")
        lines.append(f"  Inputs: {struct['num_inputs']}")
        lines.append(f"  Outputs: {struct['num_outputs']}")

        if "op_type_counts" in struct:
            lines.append("\n  Operation Types:")
            for op_type, count in sorted(struct["op_type_counts"].items()):
                lines.append(f"    {op_type}: {count}")

    # Validation
    if "validation" in analysis:
        val = analysis["validation"]
        lines.append("\nVALIDATION:")
        lines.append(f"  Valid: {val['is_valid']}")
        lines.append(f"  ONNX Checker: {val['onnx_checker']['valid']}")
        lines.append(f"  Runtime Load: {val['runtime']['can_load']}")
        lines.append(f"  Dead Nodes: {len(val['dead_nodes'])}")
        lines.append(f"  Broken Connections: {len(val['broken_connections'])}")
        lines.append(f"  Orphan Initializers: {len(val['orphan_initializers'])}")

        if val["dead_nodes"]:
            lines.append(f"    Dead nodes: {', '.join(val['dead_nodes'])}")

    # Patterns
    if "patterns" in analysis:
        lines.append("\nPATTERNS DETECTED:")
        for pattern_name, pattern_info in analysis["patterns"].items():
            count = pattern_info.get("count", 0)
            desc = pattern_info.get("description", pattern_name)
            lines.append(f"  {desc}: {count}")

    # Optimization opportunities
    if "optimization_opportunities" in analysis:
        opp = analysis["optimization_opportunities"]
        lines.append("\nOPTIMIZATION OPPORTUNITIES:")
        lines.append(f"  Fusible patterns: {opp['total_fusible']}")
        lines.append(f"  Redundant operations: {opp['total_redundant']}")
        lines.append(f"  Estimated node reduction: {opp['estimated_reduction']}")

    lines.append("=" * 80)
    return "\n".join(lines)


def generate_json_report(
    analysis: dict,
    output_path: str,
) -> None:
    """Generate JSON report for programmatic use.

    :param analysis: Analysis results dictionary
    :param output_path: JSON output path
    """
    with Path(output_path).open("w") as f:
        json.dump(analysis, f, indent=2)
