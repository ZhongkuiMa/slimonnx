__docformat__ = ["restructuredtext"]
__all__ = ["_shape_to_initializer"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils


def _shape_to_initializer(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int]],
    verbose: bool = False,
) -> list[NodeProto]:
    """
    Trace the shape node and make it as a direct constant.
    """

    """
    Currently, there are the following cases:
    (1) We extract the shape to construct a constant tensor. We can make such 
        constant tensor as a frozen initializer.
    (2) We extract the shape to reshape a tensor. We can make such shape as a freezed
        initializer.
    """

    nodes_dict = {node.output[0]: node for node in nodes}

    nodes_to_delete = []
    # Iterate over value_info to print tensor names and their shapes
    for node in nodes:
        if verbose:
            print(f"Node: {node.op_type} {node.name}")

        op_type = node.op_type

        if op_type == "Shape":
            value = np.array(shapes[node.output[0]])
            if len(value) == 1 and value[0] == 0:
                # This is a dynamic shape, we do not need to convert it to a constant.
                continue
            initializer = onnx.numpy_helper.from_array(value, name=node.output[0])
            initializers[node.output[0]] = initializer
            nodes_to_delete.append(node.output[0])

            if verbose:
                print(f"  Delete node: {node.name}")

            continue

        if any(
            input_name not in initializers and input_name not in nodes_to_delete
            for input_name in node.input
        ):
            # If any input is not an initializer and not a node to delete,
            # we cannot convert the node to an initializer.
            continue

        """
        NOTE: The key ideas are
        (1) Make all constants be initializers.
        (2) If all inputs are initializers: 
            (a) make itself an initializer;
            (b) delete the input initializers.
        """

        if op_type in {"Gather", "Slice", "Unsqueeze"}:
            pre_node_type = nodes_dict[node.input[0]].op_type
            if pre_node_type == "Shape":
                value = np.array(shapes[node.output[0]])
            else:
                raise NotImplementedError

        elif op_type == "Cast":
            value = onnx.numpy_helper.to_array(initializers[node.input[0]])
        elif op_type == "Reshape":
            data = onnx.numpy_helper.to_array(initializers[node.input[0]])
            shape = onnx.numpy_helper.to_array(initializers[node.input[1]])
            value = data.reshape(shape)
        elif op_type == "Range":
            start = onnx.numpy_helper.to_array(initializers[node.input[0]])
            limit = onnx.numpy_helper.to_array(initializers[node.input[1]])
            delta = onnx.numpy_helper.to_array(initializers[node.input[2]])
            value = np.arange(start, limit, delta)

        elif op_type == "ConstantOfShape":
            # The node create a constant with specified shape.
            shape = shapes[node.output[0]]
            value = onnx.numpy_helper.to_array(node.attribute[0].t)[0]
            value = np.full(shape, value, dtype=value.dtype)

        elif op_type in {"Add", "Sub", "Mul", "Div", "Concat"}:
            # Some operations have all constant inputs.
            are_initializer_inputs = all(ipt in initializers for ipt in node.input)
            if not are_initializer_inputs:
                continue

            value = None
            if op_type in {"Add", "Sub", "Mul", "Div"}:
                tensor1 = onnx.numpy_helper.to_array(initializers[node.input[0]])
                tensor2 = onnx.numpy_helper.to_array(initializers[node.input[1]])
                if op_type == "Add":
                    value = tensor1 + tensor2
                elif op_type == "Sub":
                    value = tensor1 - tensor2
                elif op_type == "Mul":
                    value = tensor1 * tensor2
                elif op_type == "Div":
                    value = tensor1 / tensor2

            elif op_type == "Concat":
                value = np.array(shapes[node.output[0]])

        else:
            raise NotImplementedError(f"Not supported node type: {op_type}.")

        initializer = onnx.numpy_helper.from_array(value, name=node.output[0])
        initializers[node.output[0]] = initializer
        nodes_to_delete.append(node.output[0])

        if verbose:
            print(f"  Delete node: {node.name}")

        for input_name in node.input:
            if input_name in initializers:
                del initializers[input_name]

    if utils.VERBOSE or verbose:
        print(f"Remove {len(nodes_to_delete)} nodes.")

    new_nodes = []
    for node in nodes:
        if len(node.output) == 1 and node.output[0] in nodes_to_delete:
            # If the node has only one output, and it is in the nodes to delete,
            # we can remove the node.
            continue
        new_nodes.append(node)

    return new_nodes
