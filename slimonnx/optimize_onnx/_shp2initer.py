__docformat__ = ["restructuredtext"]
__all__ = ["_shape_to_initializer"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto


def _shape_to_initializer(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int]],
) -> list[NodeProto]:
    """
    Trace the shape node and make it as a direct constant.
    """

    """
    Currently, there are the following cases:
    (1) We extract the shape to construct a constant tensor. we can make such 
        constant tensor as a freezed initializer.
    (2) We extract the shape to reshape a tensor. We can make such shape as a freezed
        initializer.
    """
    nodes_to_delete = []
    # Iterate over value_info to print tensor names and their shapes
    for node in nodes:
        op_type = node.op_type
        if op_type == "Shape":
            # The shape node must be deleted.
            value = np.array(shapes[node.output[0]])
            initializer = onnx.numpy_helper.from_array(value, name=node.output[0])
            initializers[node.output[0]] = initializer
            nodes_to_delete.append(node.output[0])

        if all(input_name not in nodes_to_delete for input_name in node.input):
            continue

        """
        NOTE: The key ideas are
        (1) Make all constants be initializers.
        (2) If all inputs are initializers: 
            (a) make itself an initializer;
            (b) delete the input initializers.
        """

        if op_type in {"Gather", "Slice", "Unsqueeze"}:
            # All these nodes extract or change the result from the shape node.
            value = np.array(shapes[node.output[0]])

        elif op_type == "Reshape":
            # The reshape node uses the shape node.
            continue
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

        for input_name in node.input:
            if input_name in initializers:
                del initializers[input_name]

    new_nodes = [node for node in nodes if node.output[0] not in nodes_to_delete]

    return new_nodes
