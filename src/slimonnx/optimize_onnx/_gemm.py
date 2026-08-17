"""Gemm node simplification and normalization."""

__docformat__ = "restructuredtext"
__all__ = ["_simplify_gemm"]

import onnx.numpy_helper
from onnx import NodeProto, TensorProto

from slimonnx.constants import DEFAULT_GEMM_ALPHA, DEFAULT_GEMM_BETA
from slimonnx.optimize_onnx._onnx_attrs import get_onnx_attrs


def _normalize_gemm_matrix_input(
    input_name: str,
    scale: float,
    should_transpose: bool,
    initializers: dict[str, TensorProto],
    unique_suffix: str,
) -> tuple[str, float, int]:
    """Normalize a single Gemm matrix input (weight or variable).

    Apply transpose and scaling to create a new initializer if needed.

    :param input_name: Name of the input tensor.

    :param scale: Scaling factor (alpha for weight, beta for bias).

    :param should_transpose: Whether to transpose the matrix.

    :param initializers: Dictionary of initializers (modified in-place).

    :param unique_suffix: Unique suffix for new initializer name.

    :return: Tuple of (new_input_name, new_scale, new_transpose_flag)
    """
    if input_name not in initializers:
        return input_name, scale, 1 if should_transpose else 0

    initializer = initializers[input_name]
    array = onnx.numpy_helper.to_array(initializer)

    if should_transpose:
        array = array.copy().T

    if scale != 1.0:
        array = array * scale
        scale = 1.0

    new_name = f"{input_name}_{unique_suffix}"
    new_initializer = onnx.numpy_helper.from_array(array, new_name)
    initializers[new_name] = new_initializer

    return new_name, scale, 0


def _normalize_gemm_bias_input(
    input_name: str,
    beta: float,
    initializers: dict[str, TensorProto],
    unique_suffix: str,
) -> tuple[str, float]:
    """Normalize Gemm bias input.

    :param input_name: Name of the bias tensor.

    :param beta: Beta scaling factor.

    :param initializers: Dictionary of initializers (modified in-place).

    :param unique_suffix: Unique suffix for new initializer name.

    :return: Tuple of (new_input_name, new_beta)
    """
    if input_name not in initializers:
        return input_name, beta

    initializer = initializers[input_name]
    array = onnx.numpy_helper.to_array(initializer)

    if beta != 1.0:
        array = array * beta
        beta = 1.0

    new_name = f"{input_name}_{unique_suffix}"
    new_initializer = onnx.numpy_helper.from_array(array, new_name)
    initializers[new_name] = new_initializer

    return new_name, beta


def _simplify_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """Simplify Gemm nodes by normalizing attributes and creating copies of shared initializers.

    This function:
    1. Absorbs alpha/beta/transA/transB into initializer values
    2. Creates unique copies of initializers for each Gemm (avoids shared state)
    3. Preserves every attribute that cannot be absorbed
    4. Removes only normalized default attributes
    5. Cleans up unused initializers

    :param nodes: List of nodes.

    :param initializers: Dictionary of initializers (modified in-place).

    :return: Simplified list of nodes
    """
    new_nodes = []
    gemm_count = 0

    for node in nodes:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        attrs = get_onnx_attrs(node, initializers)
        alpha = attrs.get("alpha", DEFAULT_GEMM_ALPHA)
        beta = attrs.get("beta", DEFAULT_GEMM_BETA)
        trans_a = attrs.get("transA", 0)
        trans_b = attrs.get("transB", 0)

        unique_suffix = str(gemm_count)

        # Normalize each input
        input_0, alpha, trans_a = _normalize_gemm_matrix_input(
            node.input[0], alpha, trans_a == 1, initializers, unique_suffix
        )

        input_1, alpha, trans_b = _normalize_gemm_matrix_input(
            node.input[1], alpha, trans_b == 1, initializers, unique_suffix
        )

        input_2, beta = (
            _normalize_gemm_bias_input(node.input[2], beta, initializers, unique_suffix)
            if len(node.input) > 2
            else (None, beta)
        )

        # Build new input list
        input_names = [input_0, input_1]
        if input_2 is not None:
            input_names.append(input_2)

        # Preserve protobuf metadata and every residual semantic attribute.
        # Matrix multiplication is not commutative, so input order is never
        # changed merely to put a runtime tensor first.
        new_node = NodeProto()
        new_node.CopyFrom(node)
        del new_node.input[:]
        new_node.input.extend(input_names)
        del new_node.attribute[:]
        residual_attrs = {
            "alpha": alpha,
            "beta": beta if input_2 is not None else 1.0,
            "transA": trans_a,
            "transB": trans_b,
        }
        for name, value in residual_attrs.items():
            if value != (1.0 if name in {"alpha", "beta"} else 0):
                new_node.attribute.append(onnx.helper.make_attribute(name, value))

        new_nodes.append(new_node)
        gemm_count += 1

    # Cleanup: Remove unused initializers
    all_input_names = {input_name for node in new_nodes for input_name in node.input}
    for name in list(initializers.keys()):
        if name not in all_input_names:
            del initializers[name]

    return new_nodes
