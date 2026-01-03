"""Graph structure validation for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = [
    "check_broken_connections",
    "check_dead_nodes",
    "check_orphan_initializers",
    "check_shape_consistency",
    "check_type_consistency",
]

from onnx import NodeProto, TensorProto, ValueInfoProto


def check_dead_nodes(
    nodes: list[NodeProto],
    outputs: list[ValueInfoProto],
) -> list[str]:
    """Find nodes not in dependency chain from outputs (dead code).

    :param nodes: Model nodes
    :param outputs: Model outputs
    :return: List of dead node names
    """
    # Build reverse dependency graph
    output_names = {out.name for out in outputs}
    node_outputs = {out: node for node in nodes for out in node.output}

    # Traverse backwards from outputs
    reachable = set(output_names)
    queue = list(output_names)

    while queue:
        current = queue.pop(0)
        if current in node_outputs:
            node = node_outputs[current]
            for inp in node.input:
                if inp not in reachable:
                    reachable.add(inp)
                    queue.append(inp)

    # Find unreachable nodes
    return [
        node.name if node.name else f"{node.op_type}_unnamed"
        for node in nodes
        if not any(out in reachable for out in node.output)
    ]


def check_broken_connections(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    inputs: list[ValueInfoProto],
) -> list[dict]:
    """Find nodes with missing input connections.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :param inputs: Model inputs
    :return: List of connection error dictionaries
    """
    available_tensors = set(initializers.keys())
    available_tensors.update(inp.name for inp in inputs)

    # Add all node outputs to available tensors
    for node in nodes:
        available_tensors.update(node.output)

    return [
        {
            "node": node.name if node.name else f"{node.op_type}_unnamed",
            "op_type": node.op_type,
            "missing_input": inp,
        }
        for node in nodes
        for inp in node.input
        if inp and inp not in available_tensors
    ]


def check_orphan_initializers(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[str]:
    """Find initializers not used by any node.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of orphan initializer names
    """
    used_initializers = set()
    for node in nodes:
        used_initializers.update(node.input)

    return [init_name for init_name in initializers if init_name not in used_initializers]


def check_type_consistency(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[dict]:
    """Check tensor type consistency across connections.

    :param nodes: Model nodes
    :param initializers: Model initializers
    :return: List of type error dictionaries
    """
    # Get tensor types from initializers
    tensor_types = {}
    for name, init in initializers.items():
        tensor_types[name] = init.data_type

    # For now, just check if initializers have valid types
    # More sophisticated type checking would require shape inference
    errors = []
    for name, dtype in tensor_types.items():
        if dtype == 0:  # UNDEFINED type
            errors.append(
                {
                    "tensor": name,
                    "error": "Undefined data type",
                }
            )

    return errors


def check_shape_consistency(
    nodes: list[NodeProto],
    data_shapes: dict[str, int | list[int]],
) -> list[dict]:
    """Check shape compatibility for operations.

    :param nodes: Model nodes
    :param data_shapes: Inferred shapes dictionary
    :return: List of shape error dictionaries
    """
    input_errors = [
        {
            "node": node.name if node.name else f"{node.op_type}_unnamed",
            "op_type": node.op_type,
            "input": inp,
            "error": "Unknown shape",
        }
        for node in nodes
        for inp in node.input
        if inp and inp not in data_shapes
    ]

    output_errors = [
        {
            "node": node.name if node.name else f"{node.op_type}_unnamed",
            "op_type": node.op_type,
            "output": out,
            "error": "Unknown shape",
        }
        for node in nodes
        for out in node.output
        if out and out not in data_shapes
    ]

    return input_errors + output_errors
