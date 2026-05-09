"""ONNX model validation utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "check_broken_connections",
    "check_dead_nodes",
    "check_orphan_initializers",
    "check_shape_consistency",
    "check_type_consistency",
    "compare_model_outputs",
    "generate_inputs_from_bounds",
    "run_onnx_checker",
    "run_onnx_inference",
    "validate_model",
    "validate_with_onnxruntime",
]

from slimonnx.model_validate._orchestrator import validate_model
from slimonnx.model_validate.graph_validator import (
    check_broken_connections,
    check_dead_nodes,
    check_orphan_initializers,
    check_shape_consistency,
    check_type_consistency,
)
from slimonnx.model_validate.numerical_compare import (
    compare_model_outputs,
    generate_inputs_from_bounds,
    run_onnx_inference,
)
from slimonnx.model_validate.onnx_checker import run_onnx_checker
from slimonnx.model_validate.runtime_validator import validate_with_onnxruntime
