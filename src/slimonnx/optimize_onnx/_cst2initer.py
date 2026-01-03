__docformat__ = "restructuredtext"
__all__ = ["_constant_to_initializer"]


import onnx
from onnx import NodeProto, TensorProto


def _constant_to_initializer(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            np_array = onnx.numpy_helper.to_array(node.attribute[0].t)
            initializer = onnx.numpy_helper.from_array(np_array, node.output[0])
            initializers[node.output[0]] = initializer

            continue

        new_nodes.append(node)

    return new_nodes
