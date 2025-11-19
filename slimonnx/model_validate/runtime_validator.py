"""ONNX Runtime validation."""

__docformat__ = "restructuredtext"
__all__ = ["validate_with_onnxruntime"]

import tempfile

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
    try:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            onnx.save(model, tmp.name)
            session = ort.InferenceSession(tmp.name, providers=["CPUExecutionProvider"])

        result = {
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
            except Exception as e:
                result["error"] = f"Inference failed: {e}"

        return result

    except Exception as e:
        return {
            "can_load": False,
            "can_infer": False,
            "error": str(e),
        }
