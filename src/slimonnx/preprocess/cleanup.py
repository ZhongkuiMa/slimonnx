"""ONNX model cleanup utilities."""

__docformat__ = "restructuredtext"
__all__ = ["cleanup_model", "clear_docstrings", "mark_slimonnx_model"]

from onnx import ModelProto


def clear_docstrings(model: ModelProto) -> ModelProto:
    """Clear all docstrings from ONNX model nodes.

    :param model: Input ONNX model
    :return: Model with cleared docstrings
    """
    for node in model.graph.node:
        node.doc_string = ""
    return model


def mark_slimonnx_model(
    model: ModelProto,
    version: str = "1.0.0",
) -> ModelProto:
    """Mark model as processed by SlimONNX.

    :param model: Input ONNX model
    :param version: SlimONNX version string
    :return: Marked model
    """
    # Mark in producer_name
    model.producer_name = f"SlimONNX-{version}"

    # Also mark in model doc_string
    model.doc_string = f"Optimized by SlimONNX v{version}"

    return model


def cleanup_model(
    model: ModelProto,
    clear_docs: bool = True,
    mark_producer: bool = True,
    slimonnx_version: str = "1.0.0",
) -> ModelProto:
    """Full cleanup pipeline for ONNX model.

    :param model: Input ONNX model
    :param clear_docs: Whether to clear node docstrings
    :param mark_producer: Whether to mark as SlimONNX
    :param slimonnx_version: SlimONNX version string
    :return: Cleaned model
    """
    if clear_docs:
        model = clear_docstrings(model)

    if mark_producer:
        model = mark_slimonnx_model(model, version=slimonnx_version)

    return model
