"""ONNX Runtime validation."""

__docformat__ = "restructuredtext"
__all__ = ["validate_with_onnxruntime"]

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto


def validate_with_onnxruntime(
    model: ModelProto,
    test_inputs: dict[str, np.ndarray] | None = None,
) -> dict:
    """Validate model with ONNX Runtime.

    :param model: ONNX model
    :param test_inputs: Optional test inputs for inference
    :return: Validation result dictionary
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "can_load": False,
            "can_infer": False,
            "error": "onnxruntime not installed",
        }

    # Try to load model
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        onnx.save(model, tmp_path)
        session = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])

        result: dict[str, Any] = {
            "can_load": True,
            "can_infer": False,
            "error": None,
        }

        # Try inference if test inputs provided
        if test_inputs is not None:
            try:
                outputs = session.run(None, test_inputs)
                result["can_infer"] = True
                result["output_shapes"] = [list(out.shape) for out in outputs]
            except (RuntimeError, ValueError, TypeError) as error:
                result["error"] = f"Inference failed: {error}"

        return result

    except (OSError, RuntimeError, ValueError) as error:
        return {
            "can_load": False,
            "can_infer": False,
            "error": str(error),
        }
    finally:
        # Clean up temp file
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError as error:
                print(f"Failed to remove temp file {tmp_path}: {error}")
