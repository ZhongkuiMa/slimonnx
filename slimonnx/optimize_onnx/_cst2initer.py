__docformat__ = ["restructuredtext"]
__all__ = ["_constant_to_initializer"]


import onnx
from onnx import NodeProto, TensorProto

from .. import utils


def _constant_to_initializer(
    nodes: list[NodeProto], initializers: dict[str, TensorProto]
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    for node in nodes:
        if node.op_type == "Constant":
            np_array = onnx.numpy_helper.to_array(node.attribute[0].t)
            initializer = onnx.numpy_helper.from_array(np_array, node.output[0])
            initializers[node.output[0]] = initializer

            count += 1
            continue

        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Convert {count} constant nodes to initializers.")

    return new_nodes
