__docformat__ = "restructuredtext"
__all__ = ["clear_onnx_docstring", "get_initializers", "get_next_nodes_mapping"]

import onnx


def clear_onnx_docstring(model: onnx.ModelProto):
    """
    Clear docstring of all nodes in the model.

    :param model: The ONNX model.
    """
    for node in model.graph.node:
        node.doc_string = ""


def get_initializers(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    """
    Get initializers of the model.

    :param model: The ONNX model.

    .. note::
        This function only return the attributes useful for neural network
        verification.

    :return: A dictionary of initializers with key is the name of the initializer.
    """
    initializers = {}
    for initializer in model.graph.initializer:
        initializers[initializer.name] = initializer
    return initializers


def get_next_nodes_mapping(nodes: list[onnx.NodeProto]) -> dict[str, list[str]]:
    name_and_output_name_mapping = {}
    for node in nodes:
        for output_name in node.output:
            name_and_output_name_mapping[output_name] = node.name

    next_nodes_mapping = {node.name: [] for node in nodes}
    for node in nodes:
        for input_name in node.input:
            if input_name in name_and_output_name_mapping:
                next_nodes_mapping[name_and_output_name_mapping[input_name]].append(
                    node.name
                )
    return next_nodes_mapping
