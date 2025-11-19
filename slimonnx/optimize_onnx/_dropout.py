"""Remove Dropout nodes for inference optimization."""

__docformat__ = "restructuredtext"
__all__ = ["remove_dropout"]

from onnx import ModelProto, NodeProto


def remove_dropout(model: ModelProto) -> ModelProto:
    """Remove all Dropout nodes from the model.

    Dropout is a training-only operation that randomly zeros elements during training.
    During inference, Dropout nodes are redundant and should be removed.

    For ONNX Dropout nodes:
    - Input: data tensor
    - Outputs: output tensor (same as input), mask tensor (optional)
    - During inference, output = input (no-op)

    This function removes Dropout nodes and reconnects the graph:
    - Input of Dropout -> Output consumers of Dropout
    - Removes Dropout node from graph

    :param model: ONNX model
    :return: Optimized model with Dropout nodes removed
    """
    graph = model.graph
    nodes = list(graph.node)

    # Build mapping of Dropout outputs to their inputs
    dropout_output_to_input = {}
    nodes_to_remove = []

    for node in nodes:
        if node.op_type == "Dropout":
            # Dropout has 1 input and 1-2 outputs (data output + optional mask output)
            if len(node.input) > 0 and len(node.output) > 0:
                dropout_input = node.input[0]
                dropout_output = node.output[0]

                # Map dropout output to its input (bypass the dropout)
                dropout_output_to_input[dropout_output] = dropout_input
                nodes_to_remove.append(node)

    # If no Dropout nodes found, return model as-is
    if not nodes_to_remove:
        return model

    # Update all nodes that consume Dropout outputs
    new_nodes = []
    for node in nodes:
        if node in nodes_to_remove:
            continue

        # Create new node with updated inputs
        new_input = []
        modified = False
        for inp in node.input:
            if inp in dropout_output_to_input:
                new_input.append(dropout_output_to_input[inp])
                modified = True
            else:
                new_input.append(inp)

        if modified:
            # Create new node with updated inputs
            new_node = NodeProto()
            new_node.CopyFrom(node)
            new_node.ClearField("input")
            new_node.input.extend(new_input)
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    # Update graph outputs if they reference Dropout outputs
    new_outputs = []
    for output in graph.output:
        if output.name in dropout_output_to_input:
            # Create new output with updated name
            from onnx import ValueInfoProto

            new_output = ValueInfoProto()
            new_output.CopyFrom(output)
            new_output.name = dropout_output_to_input[output.name]
            new_outputs.append(new_output)
        else:
            new_outputs.append(output)

    # Rebuild graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    if new_outputs:
        graph.ClearField("output")
        graph.output.extend(new_outputs)

    return model
