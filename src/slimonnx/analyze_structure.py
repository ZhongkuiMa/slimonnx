"""Complete ONNX model analysis orchestrator."""

__docformat__ = "restructuredtext"
__all__ = ["analyze_model", "compare_models"]

import sys
from pathlib import Path

# Add parent to path for shapeonnx import
sys.path.insert(0, str(Path(__file__).parent.parent))

from shapeonnx.infer_shape import infer_onnx_shape

from slimonnx import utils
from slimonnx.model_validate import validate_model
from slimonnx.pattern_detect import detect_all_patterns
from slimonnx.preprocess import load_and_preprocess
from slimonnx.structure_analysis import (
    analyze_structure,
    export_topology_json,
    generate_json_report,
)


def analyze_model(
    onnx_path: str,
    target_opset: int | None = None,
    export_json: bool = False,
    json_output_path: str | None = None,
    export_topology: bool = False,
    topology_output_path: str | None = None,
    has_batch_dim: bool = True,
) -> dict:
    """Complete model analysis pipeline.

    :param onnx_path: Path to ONNX model
    :param target_opset: Target opset for conversion (None = keep original)
    :param export_json: Export full analysis as JSON
    :param json_output_path: JSON output path
    :param export_topology: Export topology as JSON
    :param topology_output_path: Topology JSON output path
    :param has_batch_dim: Whether model has batch dimension
    :return: Complete analysis report dictionary
    """
    # Preprocess
    model = load_and_preprocess(
        onnx_path,
        target_opset=target_opset,
        infer_shapes=True,
        check_model=True,
    )

    original_opset = model.opset_import[0].version if model.opset_import else 0
    original_ir = model.ir_version

    # Infer shapes using shapeonnx
    initializers = utils.get_initializers(model)
    input_nodes = utils.get_input_nodes(model, initializers, has_batch_dim=has_batch_dim)
    output_nodes = utils.get_output_nodes(model, has_batch_dim=has_batch_dim)
    nodes = list(model.graph.node)

    try:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim=has_batch_dim
        )
        shape_inference_success = True
    except (ImportError, ValueError, AttributeError, KeyError, RuntimeError) as error:
        print(f"Shape inference failed: {error}")
        data_shapes = None
        shape_inference_success = False

    # Validate
    validation = validate_model(model, data_shapes=data_shapes)

    # Detect patterns
    patterns = detect_all_patterns(nodes, initializers, data_shapes)

    # Analyze structure
    structure = analyze_structure(model, data_shapes)

    # Calculate optimization opportunities
    total_fusible = sum(p["count"] for p in patterns.values() if p["category"] == "fusion")
    total_redundant = sum(p["count"] for p in patterns.values() if p["category"] == "redundant")

    # Build complete report
    report = {
        "model_path": onnx_path,
        "preprocessing": {
            "original_opset": original_opset,
            "original_ir_version": original_ir,
            "shape_inference": "success" if shape_inference_success else "failed",
        },
        "validation": validation,
        "patterns": patterns,
        "structure": structure,
        "optimization_opportunities": {
            "total_fusible": total_fusible,
            "total_redundant": total_redundant,
            "estimated_reduction": total_fusible + total_redundant,
        },
    }

    # Export topology JSON if requested
    if export_topology:
        topo_path = topology_output_path or onnx_path.replace(".onnx", "_topology.json")
        export_topology_json(nodes, topo_path, data_shapes)

    # Export full analysis JSON if requested
    if export_json:
        json_path = json_output_path or onnx_path.replace(".onnx", "_analysis.json")
        generate_json_report(report, json_path)

    return report


def compare_models(
    original_path: str,
    optimized_path: str,
) -> dict:
    """Compare original and optimized models.

    :param original_path: Original model path
    :param optimized_path: Optimized model path
    :return: Comparison report dictionary
    """
    # Analyze both models
    original_report = analyze_model(original_path)
    optimized_report = analyze_model(optimized_path)

    # Calculate differences
    original_patterns = original_report["patterns"]
    optimized_patterns = optimized_report["patterns"]

    patterns_resolved = {}
    for pattern_name in original_patterns:
        original_count = original_patterns[pattern_name]["count"]
        optimized_count = optimized_patterns[pattern_name]["count"]
        resolved = original_count - optimized_count
        if resolved != 0:
            patterns_resolved[pattern_name] = {
                "original": original_count,
                "optimized": optimized_count,
                "resolved": resolved,
            }

    # Node count comparison
    original_nodes = original_report["structure"]["node_count"]
    optimized_nodes = optimized_report["structure"]["node_count"]
    node_reduction = original_nodes - optimized_nodes
    node_reduction_pct = (node_reduction / original_nodes * 100) if original_nodes > 0 else 0

    return {
        "original": original_report,
        "optimized": optimized_report,
        "changes": {
            "node_reduction": node_reduction,
            "node_reduction_pct": node_reduction_pct,
            "patterns_resolved": patterns_resolved,
        },
    }
