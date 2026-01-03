"""Redundant operation detection."""

__docformat__ = "restructuredtext"
__all__ = [
    "detect_add_zero",
    "detect_div_one",
    "detect_identity_reshape",
    "detect_mul_one",
    "detect_pad_zero",
    "detect_sub_zero",
]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def detect_add_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Add operations with zero constant.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Add":
            continue

        # Check if one input is a zero initializer
        for inp in node.input:
            if inp in initializers:
                tensor = initializers[inp]
                array = onnx.numpy_helper.to_array(tensor)
                if np.all(array == 0):
                    instances.append(
                        {
                            "node": node.name if node.name else f"Add_{i}",
                            "initializer": inp,
                            "shape": list(array.shape),
                        }
                    )
                    break

    return instances


def detect_sub_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Sub operations with zero constant.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Sub":
            continue

        # Check if second input is a zero initializer
        if len(node.input) >= 2 and node.input[1] in initializers:
            tensor = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(tensor)
            if np.all(array == 0):
                instances.append(
                    {
                        "node": node.name if node.name else f"Sub_{i}",
                        "initializer": node.input[1],
                        "shape": list(array.shape),
                    }
                )

    return instances


def detect_mul_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Mul operations with one constant.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Mul":
            continue

        # Check if one input is a one initializer
        for inp in node.input:
            if inp in initializers:
                tensor = initializers[inp]
                array = onnx.numpy_helper.to_array(tensor)
                if np.all(array == 1):
                    instances.append(
                        {
                            "node": node.name if node.name else f"Mul_{i}",
                            "initializer": inp,
                            "shape": list(array.shape),
                        }
                    )
                    break

    return instances


def detect_div_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Div operations with one constant.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Div":
            continue

        # Check if second input is a one initializer
        if len(node.input) >= 2 and node.input[1] in initializers:
            tensor = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(tensor)
            if np.all(array == 1):
                instances.append(
                    {
                        "node": node.name if node.name else f"Div_{i}",
                        "initializer": node.input[1],
                        "shape": list(array.shape),
                    }
                )

    return instances


def detect_pad_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Pad operations with zero padding.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type != "Pad":
            continue

        # Check if padding is all zeros
        if len(node.input) >= 2 and node.input[1] in initializers:
            tensor = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(tensor)
            if np.all(array == 0):
                instances.append(
                    {
                        "node": node.name if node.name else f"Pad_{i}",
                        "pads": list(array),
                    }
                )

    return instances


def detect_identity_reshape(
    nodes: list[NodeProto],
    data_shapes: dict[str, int | list[int]],
) -> list[dict]:
    """Detect Reshape with same input/output shape.

    :param nodes: Model nodes
    :param data_shapes: Inferred shapes
    :return: List of detected instances
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type not in {"Reshape", "Flatten"}:
            continue

        # Check if input and output shapes match
        if len(node.input) > 0 and len(node.output) > 0:
            input_name = node.input[0]
            output_name = node.output[0]

            if input_name in data_shapes and output_name in data_shapes:
                input_shape = data_shapes[input_name]
                output_shape = data_shapes[output_name]

                if input_shape == output_shape:
                    instances.append(
                        {
                            "node": node.name if node.name else f"{node.op_type}_{i}",
                            "op_type": node.op_type,
                            "shape": input_shape,
                        }
                    )

    return instances
