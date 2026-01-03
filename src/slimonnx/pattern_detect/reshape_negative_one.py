"""Reshape with negative one shape detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_reshape_with_negative_one"]

import onnx
from onnx import NodeProto, TensorProto


def _ensure_shape_is_list(shape: int | list[int]) -> list[int]:
    """Convert scalar shape to list if needed."""
    return [shape] if isinstance(shape, int) else shape


def detect_reshape_with_negative_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]] | None,
) -> list[dict]:
    """Detect Reshape operations with -1 in shape tensor that can be resolved.

    A Reshape can be resolved when:
    1. The shape tensor (second input) contains -1
    2. The output shape is known from shape inference (no zeros/unknowns)

    :param nodes: Model nodes
    :param initializers: Model initializers
    :param data_shapes: Inferred shapes (required for detection)
    :return: List of detected pattern instances
    """
    if data_shapes is None:
        return []

    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Reshape":
            continue

        if len(node.input) < 2:
            continue

        shape_input_name = node.input[1]

        # Check if shape is an initializer
        if shape_input_name not in initializers:
            continue

        # Get the shape tensor values
        shape_tensor = initializers[shape_input_name]
        shape_values = onnx.numpy_helper.to_array(shape_tensor).tolist()

        # Check if -1 is present in shape
        if -1 not in shape_values:
            continue

        # Check if output shape is known (no zeros indicating unknown)
        output_name = node.output[0]
        if output_name not in data_shapes:
            continue

        output_shape = _ensure_shape_is_list(data_shapes[output_name])
        if 0 in output_shape:
            # Output shape is dynamic/unknown
            continue

        # This reshape has -1 and can be resolved
        instances.append(
            {
                "node": node.name if node.name else f"Reshape_{i}",
                "original_shape": shape_values,
                "resolved_shape": output_shape,
                "shape_tensor_name": shape_input_name,
            }
        )

    return instances
