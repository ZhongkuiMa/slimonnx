__docformat__ = "restructuredtext"
__all__ = ["_simplify_names"]

import onnx
from onnx import NodeProto, TensorProto


def _rename_io_nodes(
    input_nodes: list[onnx.ValueInfoProto],
    output_nodes: list[onnx.ValueInfoProto],
    start_counter: int,
) -> tuple[dict[str, str], dict[str, str], int]:
    """Rename input and output nodes.

    :param input_nodes: List of input nodes
    :param output_nodes: List of output nodes
    :param start_counter: Starting counter value
    :return: Tuple of (node_output_names_mapping, output_old_new_mapping, counter)
    """
    node_output_names_mapping = {}
    counter = start_counter

    for node in input_nodes:
        new_name = f"input_{counter}"
        node_output_names_mapping[node.name] = new_name
        node.name = new_name
        counter += 1

    output_old_new_mapping = {}
    for node in output_nodes:
        new_name = f"output_{counter}"
        output_old_new_mapping[node.name] = new_name
        node.name = new_name
        counter += 1

    return node_output_names_mapping, output_old_new_mapping, counter


def _update_node_output_names(
    nodes: list[NodeProto],
    output_old_new_mapping: dict[str, str],
    node_output_names_mapping: dict[str, str],
    start_counter: int,
) -> int:
    """Update node output names.

    :param nodes: List of nodes
    :param output_old_new_mapping: Mapping of old to new output names
    :param node_output_names_mapping: Mapping of node output names
    :param start_counter: Starting counter value
    :return: Updated counter value
    """
    counter = start_counter

    for node in nodes:
        op_type = str(node.op_type)
        node.name = f"{op_type}_{counter}"
        counter += 1

        new_output_names = []
        for idx, output_name in enumerate(node.output):
            if output_name in output_old_new_mapping:
                new_output_name = output_old_new_mapping[output_name]
            else:
                new_output_name = f"{node.name}_{idx}" if len(node.output) > 1 else node.name
            new_output_names.append(new_output_name)
            node_output_names_mapping[output_name] = new_output_name

        node.output[:] = new_output_names

    return counter


def _update_node_input_names(
    nodes: list[NodeProto], node_output_names_mapping: dict[str, str]
) -> None:
    """Update node input names.

    :param nodes: List of nodes
    :param node_output_names_mapping: Mapping of node output names
    """
    for node in nodes:
        new_input_names = []
        for input_name in node.input:
            if input_name in node_output_names_mapping:
                new_input_names.append(node_output_names_mapping[input_name])
            else:
                new_input_names.append(input_name)
        node.input[:] = new_input_names


def _rename_initializers(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> dict[str, TensorProto]:
    """Rename initializers and update node references.

    :param nodes: List of nodes
    :param initializers: Dictionary of initializers
    :return: New dictionary of renamed initializers
    """
    new_initializers = {}
    initializers_name_mapping = {}

    for counter, (name, initializer) in enumerate(initializers.items()):
        new_name = f"Initializer_{counter}"
        new_initializers[new_name] = initializer
        initializer.name = new_name
        initializers_name_mapping[name] = new_name

    for node in nodes:
        for idx, input_name in enumerate(node.input):
            if input_name in initializers_name_mapping:
                node.input[idx] = initializers_name_mapping[input_name]

    return new_initializers


def _simplify_names(
    input_nodes: list[onnx.ValueInfoProto],
    output_nodes: list[onnx.ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Simplify the names of the nodes and initializers."""
    node_output_names_mapping, output_old_new_mapping, counter = _rename_io_nodes(
        input_nodes, output_nodes, 0
    )

    _update_node_output_names(nodes, output_old_new_mapping, node_output_names_mapping, counter)
    _update_node_input_names(nodes, node_output_names_mapping)
    new_initializers = _rename_initializers(nodes, initializers)

    return nodes, new_initializers
