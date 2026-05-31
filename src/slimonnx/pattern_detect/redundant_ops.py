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


def _check_constant_input(
    initializers: dict[str, TensorProto],
    input_name: str,
    target_value: float,
) -> np.ndarray | None:
    """Return the constant array if input is an initializer and all-equal-to-target.

    The all-equal check is exact, not ``np.allclose``: an Add+near-zero
    initializer is intentionally not considered a no-op because rounding
    the result would visibly change downstream computation.

    :param initializers: Initializer map keyed by tensor name.

    :param input_name: Name of the node input to check.

    :param target_value: Expected element value (0 for additive identities,
        1 for multiplicative identities, etc.).

    :return: The constant ``np.ndarray`` on match, else ``None``.
    """
    if input_name not in initializers:
        return None
    array = onnx.numpy_helper.to_array(initializers[input_name])
    if not np.all(array == target_value):
        return None
    return array


def _detect_binary_constant_value(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    op_type: str,
    target_value: float,
    *,
    commutative: bool,
) -> list[dict]:
    """Detect ``op_type`` nodes with a constant input equal to ``target_value``.

    Shared scanner for the four binary-arithmetic identity detectors (Add+0,
    Sub-0, Mul*1, Div/1). For commutative ops both inputs are scanned; for
    non-commutative ops only ``input[1]`` is considered (matching ``a - 0 == a``
    and ``a / 1 == a`` semantics).

    :param nodes: Model nodes.

    :param initializers: Initializer map keyed by tensor name.

    :param op_type: ONNX op_type literal to filter on.

    :param target_value: Value the constant operand must equal everywhere.

    :param commutative: If ``True`` scan all inputs; otherwise only ``input[1]``.

    :return: List of match dicts, one per node, with keys ``node``,
        ``initializer``, and ``shape``.
    """
    instances: list[dict] = []
    for i, node in enumerate(nodes):
        if node.op_type != op_type:
            continue

        if commutative:
            candidate_inputs: list[str] = list(node.input)
        else:
            if len(node.input) < 2:
                continue
            candidate_inputs = [node.input[1]]

        for inp in candidate_inputs:
            array = _check_constant_input(initializers, inp, target_value)
            if array is None:
                continue
            instances.append(
                {
                    "node": node.name or f"{op_type}_{i}",
                    "initializer": inp,
                    "shape": list(array.shape),
                }
            )
            break

    return instances


def detect_add_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Add operations with zero constant.

    :param nodes: Model nodes.

    :param initializers: Model initializers.

    :return: List of detected instances.
    """
    return _detect_binary_constant_value(nodes, initializers, "Add", 0, commutative=True)


def detect_sub_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Sub operations with zero constant.

    :param nodes: Model nodes.

    :param initializers: Model initializers.

    :return: List of detected instances.
    """
    return _detect_binary_constant_value(nodes, initializers, "Sub", 0, commutative=False)


def detect_mul_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Mul operations with one constant.

    :param nodes: Model nodes.

    :param initializers: Model initializers.

    :return: List of detected instances.
    """
    return _detect_binary_constant_value(nodes, initializers, "Mul", 1, commutative=True)


def detect_div_one(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Div operations with one constant.

    :param nodes: Model nodes.

    :param initializers: Model initializers.

    :return: List of detected instances.
    """
    return _detect_binary_constant_value(nodes, initializers, "Div", 1, commutative=False)


def detect_pad_zero(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect Pad operations with zero padding.

    Pad uses ``pads`` (the actual integer list) instead of ``initializer`` +
    ``shape`` because callers downstream of the detection report act on the
    pad amounts directly.

    :param nodes: Model nodes.

    :param initializers: Model initializers.

    :return: List of detected instances.
    """
    instances: list[dict] = []
    for i, node in enumerate(nodes):
        if node.op_type != "Pad":
            continue
        if len(node.input) < 2:
            continue
        array = _check_constant_input(initializers, node.input[1], 0)
        if array is None:
            continue
        instances.append(
            {
                "node": node.name or f"Pad_{i}",
                "pads": list(array),
            }
        )
    return instances


def detect_identity_reshape(
    nodes: list[NodeProto],
    data_shapes: dict[str, list[int]],
) -> list[dict]:
    """Detect Reshape with same input/output shape.

    :param nodes: Model nodes.

    :param data_shapes: Inferred shapes.

    :return: List of detected instances.
    """
    instances = []

    for i, node in enumerate(nodes):
        if node.op_type not in {"Reshape", "Flatten"}:
            continue

        if len(node.input) > 0 and len(node.output) > 0:
            input_name = node.input[0]
            output_name = node.output[0]

            if input_name in data_shapes and output_name in data_shapes:
                input_shape = data_shapes[input_name]
                output_shape = data_shapes[output_name]

                if input_shape == output_shape:
                    instances.append(
                        {
                            "node": node.name or f"{node.op_type}_{i}",
                            "op_type": node.op_type,
                            "shape": input_shape,
                        }
                    )

    return instances
