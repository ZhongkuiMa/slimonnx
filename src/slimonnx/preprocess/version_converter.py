"""ONNX version conversion utilities."""

__docformat__ = "restructuredtext"
__all__ = ["convert_model_version", "load_and_preprocess"]

import warnings

import onnx
from onnx import ModelProto, version_converter

from slimonnx import utils

# Recommended opset range based on shapeonnx compatibility testing
RECOMMENDED_OPSET = 20
MIN_TESTED_OPSET = 17  # Lowest working version for maximum compatibility
MAX_TESTED_OPSET = 21  # Highest tested working version
SLIMONNX_VERSION = "2026.1.0"


def convert_model_version(
    model: ModelProto,
    target_opset: int = RECOMMENDED_OPSET,
    warn_on_diff: bool = True,
) -> ModelProto:
    """Convert ONNX model to specified opset version.

    :param model: Input ONNX model
    :param target_opset: Target opset version
    :param warn_on_diff: Warn if target differs from recommended
    :return: Converted model (IR version set automatically by ONNX)
    """
    current_opset = model.opset_import[0].version if model.opset_import else 0

    # Warn if target_opset is outside recommended range
    if warn_on_diff and not (MIN_TESTED_OPSET <= target_opset <= MAX_TESTED_OPSET):
        warnings.warn(
            f"Target opset {target_opset} is outside tested range "
            f"[{MIN_TESTED_OPSET}, {MAX_TESTED_OPSET}]. "
            f"Recommended opset is {RECOMMENDED_OPSET} for maximum compatibility.",
            UserWarning,
            stacklevel=2,
        )

    # Convert if different
    if current_opset != target_opset:
        try:
            model = version_converter.convert_version(model, target_opset)
        except (ValueError, RuntimeError, AttributeError) as error:
            warnings.warn(
                f"Version conversion failed from opset {current_opset} to {target_opset}: {error}. "
                f"Keeping original opset version.",
                UserWarning,
                stacklevel=2,
            )

    return model


def load_and_preprocess(
    onnx_path: str,
    target_opset: int | None = None,
    infer_shapes: bool = True,
    check_model: bool = True,
    clear_docstrings: bool = True,
    mark_slimonnx: bool = True,
) -> ModelProto:
    """Load ONNX model and preprocess for SlimONNX.

    Preprocessing steps:
    1. Load model from file
    2. Validate with ONNX checker (if enabled)
    3. Convert to target opset version (if specified)
    4. Run shape inference (if enabled)
    5. Clear node docstrings (if enabled)
    6. Mark as processed by SlimONNX (if enabled)

    :param onnx_path: Path to ONNX file
    :param target_opset: Target opset version (None = keep original)
    :param infer_shapes: Whether to run shape inference
    :param check_model: Whether to validate model with onnx.checker
    :param clear_docstrings: Whether to clear node docstrings
    :param mark_slimonnx: Whether to mark model as processed by SlimONNX
    :return: Preprocessed model
    """
    # Load model
    model = onnx.load(onnx_path)

    # Check model validity
    if check_model:
        try:
            onnx.checker.check_model(model)
        except (ValueError, AttributeError, TypeError) as error:
            raise ValueError(f"Invalid ONNX model: {error}") from error

    # Convert opset version if requested
    if target_opset is not None:
        model = convert_model_version(model, target_opset=target_opset)

    # Infer shapes if requested
    if infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except (ValueError, RuntimeError, AttributeError) as error:
            warnings.warn(f"Shape inference failed: {error}", UserWarning, stacklevel=2)

    # Clear docstrings
    if clear_docstrings:
        model = utils.clear_onnx_docstring(model)

    # Mark as SlimONNX processed
    if mark_slimonnx:
        model.producer_name = f"SlimONNX-{SLIMONNX_VERSION}"
        model.doc_string = f"Processed by SlimONNX v{SLIMONNX_VERSION}"

    return model
