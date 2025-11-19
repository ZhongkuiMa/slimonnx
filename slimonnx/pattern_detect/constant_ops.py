"""Constant folding pattern detection."""

__docformat__ = "restructuredtext"
__all__ = ["detect_constant_foldable"]

from onnx import NodeProto, TensorProto

# Operations supported by constant folding (from optimize_onnx/_cst_op.py)
FOLDABLE_OP_TYPES = {
    "Shape",
    "Gather",
    "Slice",
    "Unsqueeze",
    "Reshape",
    "Range",
    "ConstantOfShape",
    "ReduceSum",
    "Concat",
    "Relu",
    "Neg",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "MatMul",
    "Pow",
    "Cast",
    "Equal",
    "Where",
    "Expand",
}


def detect_constant_foldable(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Detect operations with all-constant inputs that can be folded at compile time.

    Constant folding evaluates constant operations during optimization rather than runtime.
    This reduces model size and improves inference speed.

    Supported operations: Shape, Gather, Slice, Unsqueeze, Reshape, Range, ConstantOfShape,
    ReduceSum, Concat, Relu, Neg, Add, Sub, Mul, Div, MatMul, Pow, Cast, Equal, Where, Expand

    :param nodes: List of ONNX nodes
    :param initializers: Dictionary of initializers
    :param data_shapes: Optional shape information (unused)
    :return: List of foldable operation instances
    """
    instances = []

    # Build set of all available constant values
    # This includes initializers and outputs of Constant nodes
    available_constants = set(initializers.keys())

    for node in nodes:
        # Add Constant node outputs to available constants
        if node.op_type == "Constant":
            available_constants.add(node.output[0])

    # Now detect foldable operations
    for node in nodes:
        # Skip Constant nodes themselves (already handled)
        if node.op_type == "Constant":
            continue

        # Check if this operation type is foldable
        if node.op_type not in FOLDABLE_OP_TYPES:
            continue

        # Check if all inputs are constants
        all_inputs_constant = all(inp in available_constants for inp in node.input)

        if not all_inputs_constant:
            continue

        instances.append(
            {
                "node": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "can_fold": True,
            }
        )

    return instances
