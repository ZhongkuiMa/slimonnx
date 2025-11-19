__docformat__ = "restructuredtext"
__all__ = ["_fuse_constant_nodes"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

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
    (2) We extract the shape to reshape a tensor. We can make such shape as a frozen
        initializer.
    """

    nodes_dict = {node.output[0]: node for node in nodes}

    nodes_to_delete = []
    # Iterate over value_info to print tensor names and their shapes
    for node in nodes:
        # if verbose:
        #     print(f"Node: {node.op_type} {node.name}")

        op_type = node.op_type

        if op_type == "Shape":
            if verbose:
                print(f"Handle Shape node {node.name}")

            value = np.array(shapes[node.output[0]])
            if len(value) == 1 and value[0] == 0:
                # This is a dynamic shape, we do not need to convert it to a constant.
                if verbose:
                    print(f"\tSkip dynamic shape for {node.output[0]}.")
                continue

            initializer = onnx.numpy_helper.from_array(value, node.output[0])
            initializers[node.output[0]] = initializer
            nodes_to_delete.append(node.output[0])
            if verbose:
                print(f"\tSave initializer {node.output[0]}: {value}")
                print(f"\tDelete node: {node.name}")

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
            if verbose:
                print(f"Handle {op_type} node {node.name}")

            pre_node_type = nodes_dict[node.input[0]].op_type
            if pre_node_type == "Shape":
                value = np.array(shapes[node.output[0]])
            else:
                tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
                if op_type == "Gather":
                    axis = get_onnx_attrs(node, initializers)["axis"]
                    indices = onnx.numpy_helper.to_array(initializers[node.input[1]])
                    indices += 1  # Ignore the batch dimension.
                    value = np.take(tensor, indices, axis=axis)
                elif op_type == "Slice":
                    starts = onnx.numpy_helper.to_array(initializers[node.input[1]])
                    ends = onnx.numpy_helper.to_array(initializers[node.input[2]])
                    axes = initializers.get(node.input[3])
                    axes += 1  # Ignore the batch dimension.
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

        elif op_type == "Reshape":
            if verbose:
                print(f"Handle Reshape node {node.name}")

            data = onnx.numpy_helper.to_array(initializers[node.input[0]])
            shape = onnx.numpy_helper.to_array(initializers[node.input[1]])
            value = data.reshape(shape)

        elif op_type == "Range":
            if verbose:
                print(f"Handle Range node {node.name}")

            start = onnx.numpy_helper.to_array(initializers[node.input[0]])
            limit = onnx.numpy_helper.to_array(initializers[node.input[1]])
            delta = onnx.numpy_helper.to_array(initializers[node.input[2]])
            value = np.arange(start, limit, delta)

        elif op_type == "ConstantOfShape":
            if verbose:
                print(f"Handle ConstantOfShape node {node.name}")

            shape = shapes[node.output[0]]
            # Here we need to remove the redundant batch dimension.
            shape = shape[1:] if len(shape) > 1 and shape[0] == 1 else shape
            value = onnx.numpy_helper.to_array(node.attribute[0].t)[0]
            value = np.full(shape, value, dtype=value.dtype)

        elif op_type == "ReduceSum":
            if verbose:
                print(f"Handle ReduceSum node {node.name}")

            tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
            attrs = get_onnx_attrs(node, initializers)
            if len(node.input[1]) > 1:
                axes = onnx.numpy_helper.to_array(initializers[node.input[1]])
                keepdims = attrs["keepdims"]
                value = np.sum(tensor, axis=tuple(axes), keepdims=keepdims)
            else:
                noop_with_empty_axes = attrs["noop_with_empty_axes"]
                if noop_with_empty_axes:
                    value = tensor
                else:
                    axes = list(range(len(tensor.shape)))
                    keepdims = attrs["keepdims"]
                    value = np.sum(tensor, axis=tuple(axes), keepdims=keepdims)

        elif op_type == "Concat":
            if verbose:
                print(f"Handle Concat node {node.name}")

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

        elif op_type in {"Relu", "Neg"}:
            if verbose:
                print(f"Handle {op_type} node {node.name}")

            tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
            if op_type == "Relu":
                value = np.maximum(tensor, 0)
            elif op_type == "Neg":
                value = -tensor

        elif op_type in {"Add", "Sub", "Mul", "Div", "MatMul", "Pow"}:
            if verbose:
                print(f"Handle {op_type} node {node.name}")

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
            elif op_type == "Pow":
                value = np.power(tensor1, tensor2)

        elif op_type == "Cast":
            if verbose:
                print(f"Handle Cast node {node.name}")

            to = get_onnx_attrs(node, initializers)["to"]
            value = onnx.numpy_helper.to_array(initializers[node.input[0]])
            if to == 1:  # float
                value = value.astype(np.float32)
            elif to == 2:  # uint8
                value = value.astype(np.uint8)
            elif to == 3:  # int8
                value = value.astype(np.int8)
            elif to == 4:  # uint16
                value = value.astype(np.uint16)
            elif to == 5:  # int16
                value = value.astype(np.int16)
            elif to == 6:  # int32
                value = value.astype(np.int32)
            elif to == 7:  # int64
                value = value.astype(np.int64)
            elif to == 8:  # string
                value = value.astype(np.str_)
            elif to == 9:  # bool
                value = value.astype(np.bool_)
            elif to == 10:  # float16
                value = value.astype(np.float16)
            elif to == 11:  # double
                value = value.astype(np.float64)
            elif to == 12:  # uint32
                value = value.astype(np.uint32)
            elif to == 13:  # uint64
                value = value.astype(np.uint64)
            elif to == 14:  # complex64
                value = value.astype(np.complex64)
            elif to == 15:  # complex128
                value = value.astype(np.complex128)
            else:
                raise NotImplementedError(f"Not supported cast type: {to}.")

        elif op_type == "Equal":
            if verbose:
                print(f"Handle Equal node {node.name}")

            tensor1 = onnx.numpy_helper.to_array(initializers[node.input[0]])
            tensor2 = onnx.numpy_helper.to_array(initializers[node.input[1]])
            value = np.equal(tensor1, tensor2)
        elif op_type == "Where":
            if verbose:
                print(f"Handle Where node {node.name}")

            condition = onnx.numpy_helper.to_array(initializers[node.input[0]])
            x = onnx.numpy_helper.to_array(initializers[node.input[1]])
            y = onnx.numpy_helper.to_array(initializers[node.input[2]])
            value = np.where(condition, x, y)
        elif op_type == "Expand":
            if verbose:
                print(f"Handle Expand node {node.name}")

            ipt = onnx.numpy_helper.to_array(initializers[node.input[0]])
            shape = shapes[node.output[0]]
            # Here we need to remove the redundant batch dimension.
            shape = shape[1:] if len(shape) > 1 and shape[0] == 1 else shape
            value = np.broadcast_to(ipt, shape)
        else:
            raise NotImplementedError(f"Not supported node type: {op_type}.")

        if verbose:
            print(f"\tSave initializer {node.output[0]}: {value}")

        initializer = onnx.numpy_helper.from_array(value, node.output[0])
        initializers[node.output[0]] = initializer
        nodes_to_delete.append(node.output[0])

        if verbose:
            print(f"\tDelete node: {node.name}")

    if verbose:
        print(f"Remove {len(nodes_to_delete)} nodes for fusing constant nodes.")

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
