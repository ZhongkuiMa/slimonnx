"""ONNX model validation utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "check_dead_nodes",
    "check_broken_connections",
    "check_orphan_initializers",
    "check_type_consistency",
    "check_shape_consistency",
    "run_onnx_checker",
    "validate_with_onnxruntime",
    "validate_model",
    "compare_model_outputs",
    "run_onnx_inference",
    "generate_inputs_from_bounds",
]

from .graph_validator import (
    check_dead_nodes,
    check_broken_connections,
    check_orphan_initializers,
    check_type_consistency,
    check_shape_consistency,
)
from .numerical_compare import (
    compare_model_outputs,
    run_onnx_inference,
    generate_inputs_from_bounds,
)
from .onnx_checker import run_onnx_checker
from .runtime_validator import validate_with_onnxruntime


def validate_model(
    model,
    data_shapes: dict[str, list[int]] | None = None,
    test_inputs: dict | None = None,
) -> dict:
    """Run all validation checks on model.

    :param model: ONNX ModelProto
    :param data_shapes: Inferred shapes for shape consistency check
    :param test_inputs: Optional test inputs for runtime validation
    :return: Validation results dictionary
    """
    from .. import utils

    nodes = list(model.graph.node)
    initializers = utils.get_initializers(model)
    inputs = list(model.graph.inp)
    outputs = list(model.graph.out)

    results = {
        "onnx_checker": run_onnx_checker(model),
        "runtime": validate_with_onnxruntime(model, test_inputs),
        "dead_nodes": check_dead_nodes(nodes, outputs),
        "broken_connections": check_broken_connections(nodes, initializers, inputs),
        "orphan_initializers": check_orphan_initializers(nodes, initializers),
        "type_errors": check_type_consistency(nodes, initializers),
        "shape_errors": [],
    }

    if data_shapes is not None:
        results["shape_errors"] = check_shape_consistency(nodes, data_shapes)

    # Calculate overall validity
    results["is_valid"] = (
        results["onnx_checker"]["valid"]
        and results["runtime"]["can_load"]
        and len(results["dead_nodes"]) == 0
        and len(results["broken_connections"]) == 0
        and len(results["type_errors"]) == 0
        and len(results["shape_errors"]) == 0
    )

    return results
