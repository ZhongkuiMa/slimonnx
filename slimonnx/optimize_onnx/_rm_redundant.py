__docformat__ = ["restructuredtext"]
__all__ = ["_remove_redundant_reshape", "_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils


def _remove_redundant_reshape(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
) -> list[NodeProto]:
    """
    Remove redundant Reshape and Flatten nodes.
    """
    count = 0

    new_nodes = []
    for node in nodes:
        if node.op_type in {"Reshape", "Flatten"}:
            # Check if the node does nothing.
            input_shape = data_shapes[node.input[0]]
            output_shape = data_shapes[node.output[0]]
            if input_shape == output_shape:
                if node.op_type == "Reshape":
                    del initializers[node.input[1]]
                # Find all nodes taking this node as input.
                # Change their input to the previous node.
                for node_j in nodes:
                    if node.output[0] in node_j.input:
                        for k in range(len(node_j.input)):
                            if node_j.input[k] == node.output[0]:
                                node_j.input[k] = node.input[0]
                count += 1
                continue

        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Remove {count} redundant Reshape/Flatten nodes.")

    return new_nodes


def _remove_redundant_operations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    """
    Remove zero adding, subtracting, multiplying, dividing operations.
    """
    count = 0

    new_nodes = []
    for node in nodes:
        if node.op_type in {"Add", "Sub"}:
            # Check if the node does nothing.
            initializer = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(initializer)
            if np.all(array == 0):
                del initializers[node.input[1]]
                # Find all nodes taking this node as input.
                # Change their input to the previous node.
                for node_j in nodes:
                    if node.output[0] in node_j.input:
                        for k in range(len(node_j.input)):
                            if node_j.input[k] == node.output[0]:
                                node_j.input[k] = node.input[0]
                count += 1
                continue
        elif node.op_type in {"Mul", "Div"}:
            # Check if the node does nothing.
            initializer = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(initializer)
            if np.all(array == 1):
                del initializers[node.input[1]]
                # Find all nodes taking this node as input.
                # Change their input to the previous node.
                for node_j in nodes:
                    if node.output[0] in node_j.input:
                        for k in range(len(node_j.input)):
                            if node_j.input[k] == node.output[0]:
                                node_j.input[k] = node.input[0]
                count += 1
                continue

        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Remove {count} redundant operations.")

    return new_nodes
