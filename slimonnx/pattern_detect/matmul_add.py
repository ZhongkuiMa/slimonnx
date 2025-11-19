"""MatMul + Add pattern detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_matmul_add"]

import onnx
from onnx import NodeProto, TensorProto


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

        # Check if one input is an initializer (bias)
        bias_input = None
        matmul_output = None
        for inp in node.input:
            if inp in initializers:
                bias_input = inp
            else:
                matmul_output = inp

        if bias_input is None or matmul_output is None:
            continue

        # Check if the other input comes from MatMul
        if matmul_output not in output_to_node:
            continue

        matmul_node = output_to_node[matmul_output]
        if matmul_node.op_type != "MatMul":
            continue

        # Check if one input of MatMul is an initializer (weight)
        weight_input = None
        for inp in matmul_node.input:
            if inp in initializers:
                weight_input = inp
                break

        if weight_input is None:
            continue

        # Get tensor dimensions
        weight = initializers[weight_input]
        weight_array = onnx.numpy_helper.to_array(weight)
        weight_dim = len(weight_array.shape)

        bias = initializers[bias_input]
        bias_array = onnx.numpy_helper.to_array(bias)
        bias_dim = len(bias_array.shape)

        # Check dimensional constraints
        can_fuse = True
        if weight_dim != 2:
            can_fuse = False
        if bias_dim != 1:
            can_fuse = False

        # Record pattern
        patterns.append(
            {
                "matmul_node": matmul_node.name if matmul_node.name else f"MatMul_{i}",
                "add_node": node.name if node.name else f"Add_{i}",
                "weight": weight_input,
                "bias": bias_input,
                "can_fuse": can_fuse,
                "weight_shape": list(weight_array.shape),
                "bias_shape": list(bias_array.shape),
            }
        )

    return patterns
