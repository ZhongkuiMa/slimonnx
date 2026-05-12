"""Top-level orchestrator that runs all model validation checks."""

__docformat__ = "restructuredtext"
__all__ = ["validate_model"]

from slimonnx.model_validate.graph_validator import (
    check_broken_connections,
    check_dead_nodes,
    check_orphan_initializers,
    check_shape_consistency,
    check_type_consistency,
)
from slimonnx.model_validate.onnx_checker import run_onnx_checker
from slimonnx.model_validate.runtime_validator import validate_with_onnxruntime
from slimonnx.utils import get_initializers


def validate_model(
    model,
    data_shapes: dict[str, int | list[int]] | None = None,
    test_inputs: dict | None = None,
) -> dict:
    """Run all validation checks on model.

    :param model: ONNX ModelProto.

    :param data_shapes: Inferred shapes for shape consistency check.

    :param test_inputs: Optional test inputs for runtime validation.

    :return: Validation results dictionary
    """
    nodes = list(model.graph.node)
    initializers = get_initializers(model)
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)

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

    is_valid_result = (
        results["onnx_checker"]["valid"]  # type: ignore[index]
        and results["runtime"]["can_load"]  # type: ignore[index]
        and len(results["dead_nodes"]) == 0
        and len(results["broken_connections"]) == 0
        and len(results["type_errors"]) == 0
        and len(results["shape_errors"]) == 0
    )
    results["is_valid"] = is_valid_result  # type: ignore[assignment]

    return results
