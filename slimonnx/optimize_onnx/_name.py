__docformat__ = ["restructuredtext"]
__all__ = ["_simplify_names"]

import onnx
from onnx import NodeProto, TensorProto


def _simplify_names(
    input_nodes: list[onnx.ValueInfoProto],
    output_nodes: list[onnx.ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """
    Simplify the names of the nodes and initializers.
    """
    node_output_names_mapping = {}

    # Change the node name of all nodes
    counter = 0
    for node in input_nodes:
        new_name = f"input_{counter}"
        node_output_names_mapping[node.name] = new_name  # For update input names
        node.name = new_name
        counter += 1

    for node in nodes:
        op_type = str(node.op_type)
        node.name = f"{op_type}_{counter}"
        counter += 1

    assert len(output_nodes) == 1
    for node in output_nodes:
        new_name = f"output_{counter}"
        node.name = new_name
        counter += 1

    # Change the node output name of all nodes
    for node in nodes:
        new_output_names = []
        for idx, output_name in enumerate(node.output):
            if len(node.output) > 1:
                # If there are more than one output, we need to name them by index
                new_output_name = f"{node.name}_{idx}"
            else:
                new_output_name = node.name
            new_output_names.append(new_output_name)
            node_output_names_mapping[output_name] = new_output_name
        # Change the original output names
        node.output[:] = new_output_names

    # Set the output name
    last_node = nodes[-1]
    last_node.output[:] = [output_nodes[0].name]

    # Change the input name of all nodes
    for node in nodes:
        new_input_names = []
        for input_name in node.input:
            if input_name in node_output_names_mapping:
                new_input_names.append(node_output_names_mapping[input_name])
            else:
                new_input_names.append(input_name)
        node.input[:] = new_input_names

    # Change the initializer name of all initializers
    # There maybe one initializer is not used by more than one node
    # So we number them dependently.
    counter = 0
    new_initializers = {}
    initializers_name_mapping = {}
    for name, initializer in initializers.items():
        new_name = f"Initializer_{counter}"
        new_initializers[new_name] = initializer
        initializer.name = new_name
        initializers_name_mapping[name] = new_name
        counter += 1

    for node in nodes:
        for idx, input_name in enumerate(node.input):
            if input_name in initializers_name_mapping:
                node.input[idx] = initializers_name_mapping[input_name]

    return nodes, new_initializers
