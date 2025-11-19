"""ONNX version conversion utilities."""

__docformat__ = "restructuredtext"
__all__ = ["convert_model_version", "load_and_preprocess"]

import onnx
from onnx import ModelProto, version_converter


def convert_model_version(
    model: ModelProto,
    target_opset: int = 17,
) -> ModelProto:
    """Convert ONNX model to specified opset version.

    :param model: Input ONNX model
    :param target_opset: Target opset version
    :return: Converted model (IR version set automatically by ONNX)
    """
    # Convert opset using ONNX's version_converter
    # IR version is automatically updated by the converter
    current_opset = model.opset_import[0].version if model.opset_import else 0
    if current_opset != target_opset:
        model = version_converter.convert_version(model, target_opset)

    return model


def load_and_preprocess(
    onnx_path: str,
    target_opset: int | None = None,
    infer_shapes: bool = True,
    check_model: bool = True,
) -> ModelProto:
    """Load ONNX model and optionally preprocess.

    :param onnx_path: Path to ONNX file
    :param target_opset: Target opset version (None = keep original, IR version auto-updated)
    :param infer_shapes: Whether to run shape inference
    :param check_model: Whether to validate model with onnx.checker
    :return: Preprocessed model
    """
    # Load model
    model = onnx.load(onnx_path)

    # Check model validity
    if check_model:
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")

    # Convert opset version if requested (IR version automatically updated)
    if target_opset is not None:
        model = convert_model_version(model, target_opset=target_opset)

    # Infer shapes if requested
    if infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            # Shape inference failure is non-fatal, just warn
            print(f"Warning: Shape inference failed: {e}")

    return model
