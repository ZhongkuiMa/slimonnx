__docformat__ = ["restructuredtext"]
__all__ = ["_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils


def _remove_redundant_operations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
) -> list[NodeProto]:
    """
    Remove zero adding, subtracting, multiplying, dividing operations.
    """
    count = 0

    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        # Check two consecutive Reshape nodes.
        # Omit the first Reshape node and directly connect the second one to the
        # previous node.
        if (
            pre_node is not None
            and node.op_type == "Reshape"
            and pre_node.op_type == "Reshape"
        ):
            assert len(pre_node.input) == 2
            assert len(pre_node.output) == 1
            assert len(node.input) == 2
            for i, output_name in enumerate(pre_pre_node.output):
                if output_name == pre_node.input[0]:
                    node.input[0] = output_name
            count += 1
            new_nodes.pop()

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    nodes = new_nodes
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
        elif node.op_type in {"Add", "Sub"}:
            # TODO: Be compatible with any input is initializer.
            # Check if the node does nothing.
            if node.input[1] not in initializers:
                new_nodes.append(node)
                continue
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
            # TODO: Be compatible with any input is initializer.
            # Check if the node does nothing.
            if node.input[1] not in initializers:
                new_nodes.append(node)
                continue
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
        elif node.op_type in {"Pad"}:
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

        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Remove {count} redundant operations.")

    return new_nodes
