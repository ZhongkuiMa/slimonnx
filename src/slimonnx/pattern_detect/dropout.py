"""Dropout detection for inference optimization."""

__docformat__ = "restructuredtext"
__all__ = ["detect_dropout"]

from onnx import NodeProto, TensorProto


def detect_dropout(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None = None,
) -> list[dict]:
    """Detect Dropout nodes that should be removed for inference.

    Dropout is a training-only operation that randomly zeros elements.
    During inference, Dropout either does nothing or scales by keep_prob.
    For inference optimization, all Dropout nodes should be removed.

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers (unused)
    :param data_shapes: Optional shape information (unused)
    :return: List of Dropout node instances
    """
    instances = []

    for node in nodes:
        if node.op_type == "Dropout":
            # Extract ratio attribute if present
            ratio = None
            for attr in node.attribute:
                if attr.name == "ratio":
                    ratio = attr.f
                    break

            instances.append(
                {
                    "node": node.name,
                    "input": node.input[0] if node.input else None,
                    "output": node.output[0] if node.output else None,
                    "ratio": ratio,
                    "should_remove": True,  # Always remove for inference
                }
            )

    return instances
