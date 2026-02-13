"""MatMul + Add pattern detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto


def _find_bias_and_matmul_output(
    add_node: NodeProto, initializers: dict[str, TensorProto]
) -> tuple[str | None, str | None]:
    """Find bias and matmul output from Add node inputs.

    :param add_node: Add node
    :param initializers: Model initializers
    :return: Tuple of (bias_input, matmul_output)
    """
    bias_input = None
    matmul_output = None
    for inp in add_node.input:
        if inp in initializers:
            bias_input = inp
        else:
            matmul_output = inp

    return bias_input, matmul_output


def _get_weight_input(matmul_node: NodeProto, initializers: dict[str, TensorProto]) -> str | None:
    """Get weight input from MatMul node.

    :param matmul_node: MatMul node
    :param initializers: Model initializers
    :return: Weight input name or None
    """
    for inp in matmul_node.input:
        if inp in initializers:
            return str(inp)
    return None


def _can_fuse_matmul_add(
    weight_input: str, bias_input: str, initializers: dict[str, TensorProto]
) -> tuple[bool, list[int], list[int]]:
    """Check if MatMul+Add can be fused.

    :param weight_input: Weight input name
    :param bias_input: Bias input name
    :param initializers: Model initializers
    :return: Tuple of (can_fuse, weight_shape, bias_shape)
    """
    weight = initializers[weight_input]
    weight_array = onnx.numpy_helper.to_array(weight)
    weight_shape = list(weight_array.shape)
    weight_dim = len(weight_shape)

    bias = initializers[bias_input]
    bias_array = onnx.numpy_helper.to_array(bias)
    bias_shape = list(bias_array.shape)
    bias_dim = len(bias_shape)

    can_fuse = weight_dim == 2 and bias_dim == 1

    return can_fuse, weight_shape, bias_shape


def detect_matmul_add(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Detect MatMul + Add patterns that can be fused to Gemm.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of detected pattern instances
    """
    # Build output-to-node mapping
    output_to_node = {}
    for node in nodes:
        for out in node.output:
            output_to_node[out] = node

    patterns = []

    for i, node in enumerate(nodes):
        if node.op_type != "Add":
            continue

        bias_input, matmul_output = _find_bias_and_matmul_output(node, initializers)

        if bias_input is None or matmul_output is None:
            continue

        # Check if the other input comes from MatMul
        if matmul_output not in output_to_node:
            continue

        matmul_node = output_to_node[matmul_output]
        if matmul_node.op_type != "MatMul":
            continue

        weight_input = _get_weight_input(matmul_node, initializers)
        if weight_input is None:
            continue

        can_fuse, weight_shape, bias_shape = _can_fuse_matmul_add(
            weight_input, bias_input, initializers
        )

        # Record pattern
        patterns.append(
            {
                "matmul_node": matmul_node.name or f"MatMul_{i}",
                "add_node": node.name or f"Add_{i}",
                "weight": weight_input,
                "bias": bias_input,
                "can_fuse": can_fuse,
                "weight_shape": weight_shape,
                "bias_shape": bias_shape,
            }
        )

    return patterns
