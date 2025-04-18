__docformat__ = "restructuredtext"
__all__ = [
    "EXTRACT_ATTR_MAP",
    "reformat_io_shape",
    "clear_onnx_docstring",
    "get_input_nodes",
    "get_output_nodes",
    "get_initializers",
    "get_next_nodes_mapping",
]

import onnx
from onnx import ModelProto, ValueInfoProto, NodeProto, TensorProto

EXTRACT_ATTR_MAP = {
    0: lambda x: None,  # UNDEFINED
    1: lambda x: x.f,  # FLOAT
    2: lambda x: x.i,  # INT
    3: lambda x: x.s.decode("utf-8"),  # STRING
    4: lambda x: onnx.numpy_helper.to_array(x.t),  # TENSOR
    5: lambda x: x.g,  # GRAPH
    6: lambda x: tuple(x.floats),  # FLOATS
    7: lambda x: tuple(x.ints),  # INTS
    8: lambda x: None,  # STRINGS
    9: lambda x: None,  # TENSORS
    10: lambda x: None,  # GRAPHS
    11: lambda x: None,  # SPARSE_TENSOR
}


def clear_onnx_docstring(model: ModelProto):
    """
    Clear docstring of all nodes in the model.

    :param model: The ONNX model.
    """
    for node in model.graph.node:
        node.doc_string = ""


def reformat_io_shape(node: ValueInfoProto) -> list[int]:
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]

    if len(shape) == 1:
        # Some model takes not batch dim.
        return shape

    # Sometimes the first dimension is symbolic, so we need to set it to 1
    shape = [1] + shape[1:]
    return shape


def get_input_nodes(model: ModelProto) -> list[ValueInfoProto]:
    initializers = [initializer.name for initializer in model.graph.initializer]
    # Exclude initializers from inputs because sometimes the initializers are also
    # included in the inputs
    nodes = []
    for input_i in model.graph.input:
        if input_i.name not in initializers:
            shape = reformat_io_shape(input_i)

            node = onnx.helper.make_tensor_value_info(
                name=input_i.name,
                elem_type=input_i.type.tensor_type.elem_type,
                shape=shape,
            )
            nodes.append(node)

    return nodes


def get_output_nodes(model: ModelProto) -> list[ValueInfoProto]:
    nodes = []
    for output_i in model.graph.output:
        shape = reformat_io_shape(output_i)
        node = onnx.helper.make_tensor_value_info(
            name=output_i.name,
            elem_type=output_i.type.tensor_type.elem_type,
            shape=shape,
        )
        nodes.append(node)

    return nodes


def get_initializers(model: ModelProto) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def get_next_nodes_mapping(nodes: list[NodeProto]) -> dict[str, list[str]]:
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
