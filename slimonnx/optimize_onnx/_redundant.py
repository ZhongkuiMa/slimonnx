__docformat__ = "restructuredtext"
__all__ = ["_remove_redundant_operations"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto, ValueInfoProto


def _remove_redundant_operations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
    output_nodes: list[ValueInfoProto],
    verbose: bool = False,
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
            if (
                len(pre_node.input) != 2
                or len(pre_node.output) != 1
                or len(node.input) != 2
            ):
                raise ValueError(
                    f"Invalid Reshape node structure: {pre_node.name} inputs={len(pre_node.input)}, outputs={len(pre_node.output)}, {node.name} inputs={len(node.input)}. Expected 2 inputs and 1 output."
                )
            for i, output_name in enumerate(pre_pre_node.output):
                if output_name == pre_node.input[0]:
                    node.input[0] = output_name
            count += 1
            new_nodes.pop()

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    def skip_node(node: NodeProto):
        # Find all nodes taking this node as input.
        # Change their input to the previous node.
        for node_j in nodes:
            if node.output[0] in node_j.input:
                for k in range(len(node_j.input)):
                    if node_j.input[k] == node.output[0]:
                        node_j.input[k] = node.input[0]
        # Handle the output nodes.
        for output_node_j in output_nodes:
            if node.output[0] == output_node_j.name:
                output_node_j.name = node.input[0]

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
                skip_node(node)
                count += 1
                continue

        elif node.op_type in {"Add", "Sub", "Mul", "Div"}:
            if node.input[1] in initializers:
                initializer_name = node.input[1]
            elif node.input[0] in initializers:
                initializer_name = node.input[0]
            else:
                new_nodes.append(node)
                continue
            initializer = initializers[initializer_name]
            array = onnx.numpy_helper.to_array(initializer)

            redundant = False
            if node.op_type in {"Add", "Sub"} and np.all(array == 0):
                redundant = True
            elif node.op_type in {"Mul", "Div"} and np.all(array == 1):
                redundant = True

            if redundant:
                del initializers[initializer_name]
                skip_node(node)
                count += 1
                continue

        elif node.op_type in {"Pad"}:
            # Check if the node does nothing.
            initializer = initializers[node.input[1]]
            array = onnx.numpy_helper.to_array(initializer)
            if np.all(array == 0):
                del initializers[node.input[1]]
                skip_node(node)
                count += 1
                continue

        new_nodes.append(node)

    if verbose:
        print(f"Remove {count} redundant operations.")

    return new_nodes
