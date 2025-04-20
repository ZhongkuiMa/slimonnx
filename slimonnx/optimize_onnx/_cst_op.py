__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_constant_nodes"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ..onnx_attrs import get_onnx_attrs


def _fuse_constant_nodes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int]],
    verbose: bool = False,
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
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
                tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
                if op_type == "Gather":
                    axis = get_onnx_attrs(node, initializers)["axis"]
                    indices = onnx.numpy_helper.to_array(initializers[node.input[1]])
                    value = np.take(tensor, indices, axis=axis)
                elif op_type == "Slice":
                    starts = onnx.numpy_helper.to_array(initializers[node.input[1]])
                    ends = onnx.numpy_helper.to_array(initializers[node.input[2]])
                    axes = initializers.get(node.input[3])
                    if axes is not None:
                        axes = onnx.numpy_helper.to_array(axes)
                    else:
                        axes = np.arange(len(starts))
                    steps = initializers.get(node.input[4])
                    if steps is not None:
                        steps = onnx.numpy_helper.to_array(steps)
                    else:
                        steps = np.ones_like(starts)
                    slices = [slice(None)] * len(starts)
                    for axis in axes:
                        slices[axis] = slice(starts[axis], ends[axis], steps[axis])
                    value = tensor[tuple(slices)]
                elif op_type == "Unsqueeze":
                    axes = onnx.numpy_helper.to_array(initializers[node.input[1]])
                    value = np.expand_dims(tensor, axis=tuple(axes))

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

        elif op_type in {"Add", "Sub", "Mul", "Div", "MatMul", "Concat"}:
            # Some operations have all constant inputs.
            are_initializer_inputs = all(ipt in initializers for ipt in node.input)
            if not are_initializer_inputs:
                continue

            value = None
            if op_type in {"Add", "Sub", "Mul", "Div", "MatMul"}:
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
                elif op_type == "MatMul":
                    value = np.matmul(tensor1, tensor2)

            elif op_type == "Concat":
                is_concat_shape = True
                for input_name in node.input:
                    if input_name not in initializers:
                        is_concat_shape = False
                        break

                if is_concat_shape:
                    value = np.array(shapes[node.output[0]])
                else:
                    tensor_list = []
                    for input_name in node.input:
                        tensor = onnx.numpy_helper.to_array(initializers[input_name])
                        tensor_list.append(tensor)
                    axis = get_onnx_attrs(node, initializers)["axis"]
                    value = np.concatenate(tensor_list, axis=axis)

            assert value is not None

        else:
            raise NotImplementedError(f"Not supported node type: {op_type}.")

        initializer = onnx.numpy_helper.from_array(value, name=node.output[0])
        initializers[node.output[0]] = initializer
        nodes_to_delete.append(node.output[0])

        if verbose:
            print(f"  Delete node: {node.name}")

    if utils.VERBOSE or verbose:
        print(f"Remove {len(nodes_to_delete)} nodes.")

    new_nodes = []
    for node in nodes:
        if len(node.output) == 1 and node.output[0] in nodes_to_delete:
            # If the node has only one output, and it is in the nodes to delete,
            # we can remove the node.
            continue
        new_nodes.append(node)
    nodes = new_nodes

    # Remove all unused initializers
    all_inputs = []
    for node in nodes:
        for input_name in node.input:
            all_inputs.append(input_name)
    new_initializers = {}
    for name, initializer in initializers.items():
        if name in all_inputs:
            new_initializers[name] = initializer
    initializers = new_initializers

    return nodes, initializers
