"""Collapse exact complementary-ReLU affine encodings."""

from __future__ import annotations

__docformat__ = "restructuredtext"
__all__ = ["_collapse_complementary_relu"]

from collections import Counter

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def _normalized_gemm_arrays(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return static weight and bias for one normalized three-input Gemm."""
    if len(node.input) != 3 or len(node.output) != 1 or node.attribute:
        return None
    weight = initializers.get(node.input[1])
    bias = initializers.get(node.input[2])
    if weight is None or bias is None:
        return None
    return onnx.numpy_helper.to_array(weight), onnx.numpy_helper.to_array(bias)


def _complementary_positive_columns(
    first_weight: np.ndarray,
    first_bias: np.ndarray,
    second_weight: np.ndarray,
    second_bias: np.ndarray,
) -> tuple[int, ...] | None:
    """Identify an exact ``ReLU(y) - ReLU(-y)`` permutation.

    Unit coefficients and bit-exact opposite affine rows are intentional.  A
    tolerance-based match could change the real-valued network used by a
    verifier, while pre-composing the trailing bias would round a new source
    constant.  Keeping the bias as a separate Add avoids both approximations.
    """
    arrays = (first_weight, first_bias, second_weight, second_bias)
    if any(not np.issubdtype(value.dtype, np.floating) for value in arrays):
        return None
    if len({value.dtype for value in arrays}) != 1:
        return None
    if any(not np.all(np.isfinite(value)) for value in arrays):
        return None
    if first_weight.ndim != 2 or first_bias.ndim != 1:
        return None
    if second_weight.ndim != 2 or second_bias.ndim != 1:
        return None

    input_width, middle_width = first_weight.shape
    del input_width
    if second_weight.shape[0] != middle_width:
        return None
    output_width = second_weight.shape[1]
    if first_bias.shape != (middle_width,) or second_bias.shape != (output_width,):
        return None
    if middle_width != 2 * output_width:
        return None

    used: set[int] = set()
    positive_columns: list[int] = []
    for output_index in range(output_width):
        coefficients = second_weight[:, output_index]
        nonzero = np.flatnonzero(coefficients)
        if len(nonzero) != 2:
            return None
        positive = [int(index) for index in nonzero if coefficients[index] == 1]
        negative = [int(index) for index in nonzero if coefficients[index] == -1]
        if len(positive) != 1 or len(negative) != 1:
            return None
        positive_index = positive[0]
        negative_index = negative[0]
        if positive_index in used or negative_index in used:
            return None
        if not np.array_equal(
            first_weight[:, positive_index],
            -first_weight[:, negative_index],
        ):
            return None
        if first_bias[positive_index] != -first_bias[negative_index]:
            return None
        used.update((positive_index, negative_index))
        positive_columns.append(positive_index)

    if len(used) != middle_width:
        return None
    return tuple(positive_columns)


def _collapse_complementary_relu(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    graph_output_names: set[str] | None = None,
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Collapse exact ``Gemm-ReLU-Gemm-ReLU`` complementary encodings.

    The input Gemm operators must already be normalized to
    ``Y = A @ B + C``.  A matched block becomes ``Gemm-Add-ReLU``: the first
    Gemm retains one affine row from every exact ``(+y, -y)`` pair and the Add
    retains the original trailing bias.  No derived bias is serialized.
    """
    consumers = Counter(name for node in nodes for name in node.input)
    graph_output_names = graph_output_names or set()
    rewritten: list[NodeProto] = []
    index = 0
    while index < len(nodes):
        if index + 3 >= len(nodes):
            rewritten.extend(nodes[index:])
            break
        first, first_relu, second, second_relu = nodes[index : index + 4]
        is_chain = (
            first.op_type == "Gemm"
            and first_relu.op_type == "Relu"
            and second.op_type == "Gemm"
            and second_relu.op_type == "Relu"
            and len(first.output) == 1
            and len(first_relu.input) == len(first_relu.output) == 1
            and len(second_relu.input) == 1
            and first_relu.input[0] == first.output[0]
            and len(second.input) >= 1
            and second.input[0] == first_relu.output[0]
            and second_relu.input[0] == second.output[0]
            and consumers[first.output[0]] == 1
            and consumers[first_relu.output[0]] == 1
            and consumers[second.output[0]] == 1
            and first.output[0] not in graph_output_names
            and first_relu.output[0] not in graph_output_names
            and consumers[first.input[1]] == 1
            and consumers[first.input[2]] == 1
            and consumers[second.input[1]] == 1
        )
        if not is_chain:
            rewritten.append(first)
            index += 1
            continue

        first_arrays = _normalized_gemm_arrays(first, initializers)
        second_arrays = _normalized_gemm_arrays(second, initializers)
        if first_arrays is None or second_arrays is None:
            rewritten.append(first)
            index += 1
            continue
        first_weight, first_bias = first_arrays
        second_weight, second_bias = second_arrays
        positive_columns = _complementary_positive_columns(
            first_weight,
            first_bias,
            second_weight,
            second_bias,
        )
        if positive_columns is None:
            rewritten.append(first)
            index += 1
            continue

        retained_weight = np.ascontiguousarray(first_weight[:, positive_columns])
        retained_bias = np.ascontiguousarray(first_bias[list(positive_columns)])
        initializers[first.input[1]] = onnx.numpy_helper.from_array(
            retained_weight,
            name=first.input[1],
        )
        initializers[first.input[2]] = onnx.numpy_helper.from_array(
            retained_bias,
            name=first.input[2],
        )
        del initializers[second.input[1]]

        add = onnx.helper.make_node(
            "Add",
            inputs=[first.output[0], second.input[2]],
            outputs=list(second.output),
            name=f"{second.name}/complementary_add",
        )
        rewritten.extend((first, add, second_relu))
        index += 4

    return rewritten, initializers
