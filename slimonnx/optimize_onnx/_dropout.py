"""Remove Dropout nodes for inference optimization."""

__docformat__ = "restructuredtext"
__all__ = ["remove_dropout"]

from onnx import ModelProto, NodeProto, ValueInfoProto


def _build_dropout_mapping(nodes: list[NodeProto]) -> tuple[dict[str, str], list[NodeProto]]:
    """Build mapping of dropout outputs to inputs and collect nodes to remove.

    :param nodes: All model nodes
    :return: Tuple of (dropout_output_to_input mapping, nodes_to_remove list)
    """
    dropout_output_to_input = {}
    nodes_to_remove = []

    for node in nodes:
        if node.op_type == "Dropout" and len(node.input) > 0 and len(node.output) > 0:
            dropout_input = node.input[0]
            dropout_output = node.output[0]

            dropout_output_to_input[dropout_output] = dropout_input
            nodes_to_remove.append(node)

    return dropout_output_to_input, nodes_to_remove


def _update_node_inputs(
    nodes: list[NodeProto],
    nodes_to_remove: list[NodeProto],
    dropout_mapping: dict[str, str],
) -> list[NodeProto]:
    """Update node inputs to bypass dropout nodes.

    :param nodes: All model nodes
    :param nodes_to_remove: Dropout nodes to remove
    :param dropout_mapping: Mapping of dropout outputs to inputs
    :return: New list of nodes with updated inputs
    """
    new_nodes = []
    for node in nodes:
        if node in nodes_to_remove:
            continue

        new_input = []
        modified = False
        for inp in node.input:
            if inp in dropout_mapping:
                new_input.append(dropout_mapping[inp])
                modified = True
            else:
                new_input.append(inp)

        if modified:
            new_node = NodeProto()
            new_node.CopyFrom(node)
            new_node.ClearField("input")
            new_node.input.extend(new_input)
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    return new_nodes


def _update_graph_outputs(
    outputs: list[ValueInfoProto], dropout_mapping: dict[str, str]
) -> list[ValueInfoProto]:
    """Update graph outputs to bypass dropout nodes.

    :param outputs: Original graph outputs
    :param dropout_mapping: Mapping of dropout outputs to inputs
    :return: New list of outputs with updated names
    """
    new_outputs = []
    for output in outputs:
        if output.name in dropout_mapping:
            new_output = ValueInfoProto()
            new_output.CopyFrom(output)
            new_output.name = dropout_mapping[output.name]
            new_outputs.append(new_output)
        else:
            new_outputs.append(output)

    return new_outputs


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

    dropout_mapping, nodes_to_remove = _build_dropout_mapping(nodes)

    if not nodes_to_remove:
        return model

    new_nodes = _update_node_inputs(nodes, nodes_to_remove, dropout_mapping)
    new_outputs = _update_graph_outputs(list(graph.output), dropout_mapping)

    # Rebuild graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    if new_outputs:
        graph.ClearField("output")
        graph.output.extend(new_outputs)

    return model
