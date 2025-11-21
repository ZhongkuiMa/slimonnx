"""ONNX built-in checker validation."""

__docformat__ = "restructuredtext"
__all__ = ["run_onnx_checker"]

import onnx
from onnx import ModelProto


def run_onnx_checker(model: ModelProto) -> dict:
    """Run onnx.checker.check_model validation.

    :param model: ONNX model
    :return: Validation result dictionary
    """
    try:
        onnx.checker.check_model(model)
        return {
            "valid": True,
            "error": None,
        }
    except (ValueError, AttributeError, TypeError) as error:
        return {
            "valid": False,
            "error": str(error),
        }
