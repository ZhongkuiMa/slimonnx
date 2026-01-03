__docformat__ = "restructuredtext"
__all__ = ["_fuse_constant_nodes"]

from typing import cast

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._constants import ONNX_DTYPE_TO_NUMPY
from slimonnx.optimize_onnx._onnx_attrs import get_onnx_attrs


def _ensure_shape_is_list(shape: int | list[int]) -> list[int]:
    """Convert scalar shape to list if needed.

    Some ONNX operations return shape as int (scalar) rather than list.
    This normalizes to always return a list.

    :param shape: Shape as int or list
    :return: Shape as list
    """
    return [shape] if isinstance(shape, int) else shape


def _handle_shape_extraction(
    node: NodeProto, shapes: dict[str, int | list[int]], nodes_dict: dict[str, NodeProto]
) -> np.ndarray | None:
    """Extract shape from shapes dict for Shape nodes.

    :param node: Node to execute
    :param shapes: Dictionary of tensor shapes
    :param nodes_dict: Dictionary mapping output names to nodes
    :return: Computed value or None if not a shape operation
    """
    if node.input[0] not in nodes_dict:
        return None

    pre_node_type = nodes_dict[node.input[0]].op_type
    if pre_node_type != "Shape":
        return None

    output_shape = shapes[node.output[0]]
    output_shape = _ensure_shape_is_list(output_shape)
    return np.array(output_shape, dtype=np.int64)


def _execute_gather(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray | None:
    """Execute Gather operation on constant inputs.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if not a Gather operation
    """
    if node.op_type != "Gather":
        return None

    if node.input[0] not in initializers or node.input[1] not in initializers:
        return None

    data = onnx.numpy_helper.to_array(initializers[node.input[0]])
    indices = onnx.numpy_helper.to_array(initializers[node.input[1]])
    axis = get_onnx_attrs(node, initializers)["axis"]

    return np.take(data, indices, axis=axis)


def _execute_slice(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray | None:
    """Execute Slice operation on constant inputs.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if not a Slice operation
    """
    if node.op_type != "Slice":
        return None

    if node.input[0] not in initializers:
        return None

    data = onnx.numpy_helper.to_array(initializers[node.input[0]])
    starts = onnx.numpy_helper.to_array(initializers[node.input[1]])
    ends = onnx.numpy_helper.to_array(initializers[node.input[2]])

    # Optional axes parameter
    if len(node.input) > 3 and node.input[3] in initializers:
        axes = onnx.numpy_helper.to_array(initializers[node.input[3]])
    else:
        axes = np.arange(len(starts))

    # Optional steps parameter
    if len(node.input) > 4 and node.input[4] in initializers:
        steps = onnx.numpy_helper.to_array(initializers[node.input[4]])
    else:
        steps = np.ones_like(starts)

    # Build slice objects
    slices = [slice(None)] * len(data.shape)
    for i, axis in enumerate(axes):
        slices[axis] = slice(starts[i], ends[i], steps[i])

    return data[tuple(slices)]


def _execute_unsqueeze(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray | None:
    """Execute Unsqueeze operation on constant inputs.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if not an Unsqueeze operation
    """
    if node.op_type != "Unsqueeze":
        return None

    if node.input[0] not in initializers:
        return None

    data = onnx.numpy_helper.to_array(initializers[node.input[0]])
    axes_array = onnx.numpy_helper.to_array(initializers[node.input[1]])

    return np.expand_dims(data, axis=tuple(axes_array))


def _execute_gather_slice_unsqueeze(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, int | list[int]],
    nodes_dict: dict[str, NodeProto],
) -> np.ndarray:
    """Execute Gather, Slice, or Unsqueeze operations.

    Dispatcher for shape manipulation operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :param nodes_dict: Dictionary mapping output names to nodes
    :return: Computed value
    """
    # Try shape extraction first
    result = _handle_shape_extraction(node, shapes, nodes_dict)
    if result is not None:
        return result

    # Try each operation type
    result = _execute_gather(node, initializers)
    if result is not None:
        return result

    result = _execute_slice(node, initializers)
    if result is not None:
        return result

    result = _execute_unsqueeze(node, initializers)
    if result is not None:
        return result

    raise NotImplementedError(f"Not supported operation in shape manipulation: {node.op_type}.")


def _execute_range(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray | None:
    """Execute Range operation.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if inputs not available
    """
    if (
        node.input[0] not in initializers
        or node.input[1] not in initializers
        or node.input[2] not in initializers
    ):
        return None

    start_array = onnx.numpy_helper.to_array(initializers[node.input[0]])
    limit_array = onnx.numpy_helper.to_array(initializers[node.input[1]])
    delta_array = onnx.numpy_helper.to_array(initializers[node.input[2]])

    start = start_array.item() if start_array.ndim == 0 or start_array.size == 1 else None
    limit = limit_array.item() if limit_array.ndim == 0 or limit_array.size == 1 else None
    delta = delta_array.item() if delta_array.ndim == 0 or delta_array.size == 1 else None

    if not (start is not None and limit is not None and delta is not None):
        return None

    return np.arange(start, limit, delta)


def _execute_reduce_sum(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray:
    """Execute ReduceSum operation.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value
    """
    tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
    attrs = get_onnx_attrs(node, initializers)

    if len(node.input) > 1:
        axes = onnx.numpy_helper.to_array(initializers[node.input[1]])
        keepdims = attrs["keepdims"]
        return cast(np.ndarray, np.sum(tensor, axis=tuple(axes), keepdims=keepdims))

    if attrs["noop_with_empty_axes"]:
        return tensor

    axes_list = np.arange(len(tensor.shape))
    keepdims = attrs["keepdims"]
    return cast(np.ndarray, np.sum(tensor, axis=tuple(axes_list), keepdims=keepdims))


def _execute_concat(
    node: NodeProto, initializers: dict[str, TensorProto], shapes: dict[str, int | list[int]]
) -> np.ndarray:
    """Execute Concat operation.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :return: Computed value
    """
    is_concat_shape = all(input_name in initializers for input_name in node.input)

    if is_concat_shape:
        output_shape = shapes[node.output[0]]
        output_shape = _ensure_shape_is_list(output_shape)
        return np.array(output_shape, dtype=np.int64)

    tensor_list = [onnx.numpy_helper.to_array(initializers[name]) for name in node.input]
    axis = get_onnx_attrs(node, initializers)["axis"]
    return np.concatenate(tensor_list, axis=axis)


def _execute_binary_arithmetic(node: NodeProto, initializers: dict[str, TensorProto]) -> np.ndarray:
    """Execute binary arithmetic operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value
    """
    tensor1 = onnx.numpy_helper.to_array(initializers[node.input[0]])
    tensor2 = onnx.numpy_helper.to_array(initializers[node.input[1]])
    op_type = node.op_type

    if op_type == "Add":
        return cast(np.ndarray, tensor1 + tensor2)
    if op_type == "Sub":
        return cast(np.ndarray, tensor1 - tensor2)
    if op_type == "Mul":
        return cast(np.ndarray, tensor1 * tensor2)
    if op_type == "Div":
        if np.issubdtype(tensor1.dtype, np.integer) and np.issubdtype(tensor2.dtype, np.integer):
            return cast(np.ndarray, tensor1 // tensor2)
        return cast(np.ndarray, tensor1 / tensor2)
    if op_type == "MatMul":
        return cast(np.ndarray, np.matmul(tensor1, tensor2))
    # Pow
    return cast(np.ndarray, np.power(tensor1, tensor2))


def _can_fold_node(
    node: NodeProto, initializers: dict[str, TensorProto], nodes_to_delete: list[str]
) -> bool:
    """Check if all node inputs are available for constant folding.

    :param node: Node to check
    :param initializers: Dictionary of initializers
    :param nodes_to_delete: List of nodes marked for deletion
    :return: True if node can be folded
    """
    return all(
        input_name in initializers or input_name in nodes_to_delete for input_name in node.input
    )


def _execute_shape_manipulation_ops(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, int | list[int]],
    nodes_dict: dict[str, NodeProto],
) -> np.ndarray | None:
    """Execute shape manipulation operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :param nodes_dict: Dictionary mapping output names to nodes
    :return: Computed value or None if not a shape manipulation op
    """
    op_type = node.op_type

    if op_type in {"Gather", "Slice", "Unsqueeze"}:
        return _execute_gather_slice_unsqueeze(node, initializers, shapes, nodes_dict)

    if op_type == "Reshape":
        data = onnx.numpy_helper.to_array(initializers[node.input[0]])
        shape = onnx.numpy_helper.to_array(initializers[node.input[1]])
        return data.reshape(shape)

    return None


def _execute_generation_ops(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> np.ndarray | None:
    """Execute generation operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if not a generation op
    """
    op_type = node.op_type

    if op_type == "Range":
        return _execute_range(node, initializers)

    if op_type == "ConstantOfShape":
        shape_tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
        shape = shape_tensor.tolist() if shape_tensor.ndim > 0 else [int(shape_tensor)]
        value = onnx.numpy_helper.to_array(node.attribute[0].t)[0]
        return np.full(shape, value, dtype=value.dtype)

    return None


def _execute_aggregation_ops(
    node: NodeProto, initializers: dict[str, TensorProto], shapes: dict[str, int | list[int]]
) -> np.ndarray | None:
    """Execute aggregation operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :return: Computed value or None if not an aggregation op
    """
    op_type = node.op_type

    if op_type == "ReduceSum":
        return _execute_reduce_sum(node, initializers)

    if op_type == "Concat":
        return _execute_concat(node, initializers, shapes)

    return None


def _execute_elementwise_ops(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> np.ndarray | None:
    """Execute element-wise operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :return: Computed value or None if not an element-wise op
    """
    op_type = node.op_type

    if op_type in {"Relu", "Neg"}:
        tensor = onnx.numpy_helper.to_array(initializers[node.input[0]])
        return np.maximum(tensor, 0) if op_type == "Relu" else -tensor

    if op_type in {"Add", "Sub", "Mul", "Div", "MatMul", "Pow"}:
        return _execute_binary_arithmetic(node, initializers)

    return None


def _execute_type_and_logic_ops(
    node: NodeProto, initializers: dict[str, TensorProto], shapes: dict[str, int | list[int]]
) -> np.ndarray | None:
    """Execute type conversion and logic operations.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :return: Computed value or None if not a type/logic op
    """
    op_type = node.op_type

    if op_type == "Cast":
        target_dtype = get_onnx_attrs(node, initializers)["to"]
        value = onnx.numpy_helper.to_array(initializers[node.input[0]])
        if target_dtype not in ONNX_DTYPE_TO_NUMPY:
            raise ValueError(f"Unsupported Cast dtype: {target_dtype}")
        return value.astype(ONNX_DTYPE_TO_NUMPY[target_dtype])

    if op_type == "Equal":
        tensor1 = onnx.numpy_helper.to_array(initializers[node.input[0]])
        tensor2 = onnx.numpy_helper.to_array(initializers[node.input[1]])
        return np.equal(tensor1, tensor2)

    if op_type == "Where":
        condition = onnx.numpy_helper.to_array(initializers[node.input[0]])
        operand_x = onnx.numpy_helper.to_array(initializers[node.input[1]])
        operand_y = onnx.numpy_helper.to_array(initializers[node.input[2]])
        return np.where(condition, operand_x, operand_y)

    if op_type == "Expand":
        ipt = onnx.numpy_helper.to_array(initializers[node.input[0]])
        shape = shapes[node.output[0]]
        shape = _ensure_shape_is_list(shape)
        return np.broadcast_to(ipt, shape)

    return None


def _execute_constant_operation(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, int | list[int]],
    nodes_dict: dict[str, NodeProto],
) -> np.ndarray | None:
    """Execute a constant operation and return the result.

    :param node: Node to execute
    :param initializers: Dictionary of initializers
    :param shapes: Dictionary of tensor shapes
    :param nodes_dict: Dictionary mapping output names to nodes
    :return: Computed value or None if cannot execute
    """
    result = _execute_shape_manipulation_ops(node, initializers, shapes, nodes_dict)
    if result is not None:
        return result

    result = _execute_generation_ops(node, initializers)
    if result is not None:
        return result

    result = _execute_aggregation_ops(node, initializers, shapes)
    if result is not None:
        return result

    result = _execute_elementwise_ops(node, initializers)
    if result is not None:
        return result

    result = _execute_type_and_logic_ops(node, initializers, shapes)
    if result is not None:
        return result

    raise NotImplementedError(f"Not supported node type: {node.op_type}.")


def _fuse_constant_nodes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    shapes: dict[str, int | list[int]],
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Trace the shape node and make it as a direct constant.

    Currently, there are the following cases:
    (1) We extract the shape to construct a constant tensor. We can make such
        constant tensor as a frozen initializer.
    (2) We extract the shape to reshape a tensor. We can make such shape as a frozen
        initializer.
    """
    nodes_dict = {node.output[0]: node for node in nodes}
    nodes_to_delete: list[str] = []

    for node in nodes:
        op_type = node.op_type
        value: np.ndarray | None = None

        if op_type == "Shape":
            value = np.array(shapes[node.output[0]], dtype=np.int64)
            if len(value) == 1 and value[0] == 0:
                continue

            initializer = onnx.numpy_helper.from_array(value, node.output[0])
            initializers[node.output[0]] = initializer
            nodes_to_delete.append(node.output[0])
            continue

        if not _can_fold_node(node, initializers, nodes_to_delete):
            continue

        try:
            value = _execute_constant_operation(node, initializers, shapes, nodes_dict)
        except (NotImplementedError, KeyError):
            continue

        if value is None:
            continue

        initializer = onnx.numpy_helper.from_array(value, node.output[0])
        initializers[node.output[0]] = initializer
        nodes_to_delete.append(node.output[0])

    new_nodes = [
        node for node in nodes if not (len(node.output) == 1 and node.output[0] in nodes_to_delete)
    ]

    all_inputs = [input_name for node in new_nodes for input_name in node.input]
    initializers = {
        name: initializer for name, initializer in initializers.items() if name in all_inputs
    }

    return new_nodes, initializers
