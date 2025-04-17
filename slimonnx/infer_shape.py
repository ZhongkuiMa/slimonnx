__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape"]

import math
import warnings
from math import ceil, floor
from typing import Any

import numpy
import numpy as np
import onnx
from onnx import ValueInfoProto, NodeProto, TensorProto

from .onnx_attrs import get_onnx_attrs

_VERBOSE = False


def _get_input_shape(input_nodes: list[ValueInfoProto], shapes: dict[str, list[int]]):
    for input_node in input_nodes:
        shape = [x.dim_value for x in input_node.type.tensor_type.shape.dim[1:]]
        shape = [1] + shape
        shape = [int(x) for x in shape]
        shapes[input_node.name] = shape
        if _VERBOSE:
            print(f"Input {input_node.name:<50} Shape={shape}")


def _get_output_shape(output_nodes: list[ValueInfoProto], shapes: dict[str, list[int]]):
    for output_node in output_nodes:
        shape = [x.dim_value for x in output_node.type.tensor_type.shape.dim[1:]]
        shape = [1] + shape
        shape = [int(x) for x in shape]
        shapes[output_node.name] = shape
        if _VERBOSE:
            print(f"Output {output_node.name:<50} Shape={shape}")


def _get_initializer_shape(
    initializers: dict[str, TensorProto], shapes: dict[str, list[int]]
):
    for initializer in initializers.values():
        shape = [int(x) for x in initializer.dims]
        shapes[initializer.name] = shape
        if _VERBOSE:
            print(f"Initializer {initializer.name:<50} Shape={shape}")


def _infer_shape_of_act(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape = shapes[node.input[0]]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_batch_norm(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape = shapes[node.input[0]]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_binary_op(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape1 = shapes[node.input[0]]
    shape2 = shapes[node.input[1]]

    # Use a broadcast mechanism to calculate the output shape
    if len(shape1) < len(shape2):
        shape1 = [1] * (len(shape2) - len(shape1)) + shape1
    elif len(shape1) > len(shape2):
        shape2 = [1] * (len(shape1) - len(shape2)) + shape2

    shape = []
    for i in range(len(shape1)):
        assert (
            shape1[i] == shape2[i] or shape1[i] == 1 or shape2[i] == 1
        ), "Shape mismatch"
        shape.append(max(shape1[i], shape2[i]))

    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_concat(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    attrs = get_onnx_attrs(node, initializers)
    axis = attrs["axis"]

    """
    There are two cases:
    1. Concatenate several shapes, e.g., shape1=[1, 48], shape2=[-1], then [1, 48, -1]
    2. Concatenate several tensor values.
    """
    is_value = any(name in explicit_shapes for name in node.input)
    input_shapes = []

    if is_value:
        # We will calculate the explicit shapes with initializers.
        # This is to concatenate several 1d lists of int by order
        # and the axis must be 0.
        for name in node.input:
            if name in initializers:
                value = onnx.numpy_helper.to_array(initializers[name])
            elif name in explicit_shapes:
                value = numpy.asarray(explicit_shapes[name])
            else:
                raise RuntimeError("Should not reach here.")
            input_shapes.append(value)
        value = np.concatenate(input_shapes, axis=axis).tolist()
        explicit_shapes[node.output[0]] = value
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")
        return

    # We will calculate the shapes of tensors
    # This is to calculate the size sum of the specified axis.
    for name in node.input:
        if name in shapes:
            value = shapes[name]
        elif name in initializers:
            value = onnx.numpy_helper.to_array(initializers[name]).tolist()
        else:
            raise RuntimeError("Should not reach here.")
        input_shapes.append(value.copy())
    shape = input_shapes[0]
    for i in range(1, len(input_shapes)):
        shape[axis] += input_shapes[i][axis]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_constant_of_shape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape = explicit_shapes[node.input[0]]
    shapes[node.output[0]] = shape.copy()  # Copy the shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_conv(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):

    attrs = get_onnx_attrs(node, initializers)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    if (
        not len(kernel_shape) == 2
        and len(dilations) == 2
        and len(pads) == 4
        and len(strides) == 2
    ):
        raise NotImplementedError

    input_shape = shapes[node.input[0]]
    weight_shape = list(initializers[node.input[1]].dims)

    # Calculate the output size
    temp1 = [pads[0] + pads[1], pads[2] + pads[3]]
    temp2 = [dilations[0] * (kernel_shape[0] - 1), dilations[1] * (kernel_shape[1] - 1)]
    output_hw = [0, 0]
    ceil_mode = False
    for i in range(2):
        output_hw[i] = (input_shape[i + 2] + temp1[i] - temp2[i] - 1) / strides[i] + 1
        output_hw[i] = ceil(output_hw[i]) if ceil_mode else floor(output_hw[i])

    shape = [input_shape[0], weight_shape[0]]
    for i in range(2):
        shape.append(output_hw[i])

    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_convtranspose(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    weight = initializers[node.input[1]]
    weight_shape = list(weight.dims)
    attrs = get_onnx_attrs(node, initializers)

    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    output_padding = attrs["output_padding"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    input_shape = shapes[node.input[0]]
    # Calculate the output size
    temp1 = [pads[0] + pads[1], pads[2] + pads[3]]
    temp2 = [dilations[0] * (kernel_shape[0] - 1), dilations[1] * (kernel_shape[1] - 1)]
    output_hw = [0, 0]
    ceil_mode = True
    for i in range(2):
        output_hw[i] = (
            (input_shape[i + 2] - 1) * strides[i]
            - temp1[i]
            + temp2[i]
            + output_padding[i]
            + 1
        )
        output_hw[i] = ceil(output_hw[i]) if ceil_mode else floor(output_hw[i])

    shape = [input_shape[0], weight_shape[1]]
    for i in range(2):
        shape.append(output_hw[i])

    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_flatten(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape = shapes[node.input[0]]
    axis = get_onnx_attrs(node, initializers)["axis"]
    shape = shape[:axis] + [int(math.prod(shape) / math.prod(shape[:axis]))]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_gather(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    axis = get_onnx_attrs(node, initializers)["axis"]
    indices = onnx.numpy_helper.to_array(initializers[node.input[1]])
    indices = np.expand_dims(indices, 0) if indices.ndim == 0 else indices
    value = explicit_shapes[node.input[0]]
    value = np.asarray(value)
    value = np.take_along_axis(value, indices, axis=axis).tolist()
    if len(value) == 1:
        value = value[0]
    explicit_shapes[node.output[0]] = value
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")


def _infer_shape_of_gemm(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    attrs = get_onnx_attrs(node, initializers)
    transA = attrs["transA"]
    transB = attrs["transB"]
    shape1 = shapes[node.input[0]]
    shape2 = shapes[node.input[1]]

    if transA:
        shape1 = shape1[:-2] + [shape1[-1], shape1[-2]]
    if transB:
        shape2 = shape2[:-2] + [shape2[-1], shape2[-2]]

    shape = shape1[:-1] + shape2[-1:]

    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_matmul(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape1 = shapes[node.input[0]]
    shape2 = shapes[node.input[1]]

    if not (len(shape1) >= 2 and len(shape2) >= 2 and shape1[-1] == shape2[-2]):
        raise NotImplementedError(
            f"Not supported {node.op_type:<20} with shape {shape1} and {shape2}"
        )

    shape = [*shape1[:-1], shape2[-1]]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_reduce(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    shape = shapes[node.input[0]].copy()
    keepdims = get_onnx_attrs(node, initializers)["keepdims"]
    axes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()
    for axis in axes:
        shape[axis] = 1 if keepdims else 0
    # Remove all 0 in shape
    shape = [x for x in shape if x != 0]
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_reshape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    data_shape = shapes[node.input[0]]

    if node.input[1] in explicit_shapes:
        shape = explicit_shapes[node.input[1]]
    else:
        shape = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()

    def _infer_reshape_shape(ori_shape: list[int], new_shape: list[int]) -> list[int]:
        """
        Infers the shape of an array after reshaping, without performing actual
        computation.

        :param ori_shape: Original shape of the array
        :param new_shape: New shape of the array
        :return: The reshaped array
        """
        inferred_shape = math.prod(ori_shape)
        inferred_idx = -1
        for idx, new_shape_i in enumerate(new_shape):
            if new_shape_i == -1:
                inferred_idx = idx
                continue
            inferred_shape = inferred_shape / new_shape_i
        if inferred_idx != -1:
            new_shape[inferred_idx] = int(inferred_shape)
        return new_shape

    shape = _infer_reshape_shape(data_shape, shape)
    shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


def _infer_shape_of_shape(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    # The shape node extract the shape of one node.
    value = shapes[node.input[0]]
    explicit_shapes[node.output[0]] = value
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")


def _infer_shape_of_slice(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    starts = initializers[node.input[1]]
    ends = initializers[node.input[2]]
    starts = onnx.numpy_helper.to_array(starts).tolist()
    ends = onnx.numpy_helper.to_array(ends).tolist()

    n_inputs = len(node.input)
    if n_inputs > 3:
        axes = initializers[node.input[3]]
        axes = onnx.numpy_helper.to_array(axes).tolist()
    else:
        axes = list(range(len(starts)))
    if n_inputs > 4:
        steps = initializers[node.input[4]]
        steps = onnx.numpy_helper.to_array(steps).tolist()
    else:
        steps = [1] * len(axes)

    # There are two cases:
    # (1) slice a shape, e.g., shape=[1, 2, 3, 4], shape[0:2:1]=[1, 2]
    # (2) slice a tensor value

    def _infer_sliced_shape(
        shape_: list[int],
        axes_: list[int],
        starts_: list[int],
        ends_: list[int],
        steps_: list[int],
    ) -> list[int]:
        new_shape = list(shape_)

        for axis, start, end, step in zip(axes_, starts_, ends_, steps_):
            size = shape_[axis]

            # Handle negative indices
            start = min(max(start + size if start < 0 else start, 0), size)
            end = min(max(end + size if end < 0 else end, 0), size)

            if step < 0:
                warnings.warn(f"Negative step ({step}) is not tested.")

            # Compute the new dimension size
            new_shape[axis] = max(
                0, (end - start + (step - (1 if step > 0 else -1))) // step
            )

        return new_shape

    # slice a tensor and we need its new shape
    shape = shapes.get(node.input[0])
    if shape is not None:
        shapes[node.output[0]] = _infer_sliced_shape(shape, axes, starts, ends, steps)
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")
        return

    # slice a shape (extracted by shape node) and we need the sliced shape
    value = explicit_shapes.get(node.input[0])
    if value is not None:
        # Commonly, the shape is a 1d list of int.
        # In such case, the node aims to extract a dimension on the specified axis.
        # e.g. I want to know the shape of the first dimension of the input tensor
        assert len(axes) == 1
        value = value[starts[0] : ends[0] : steps[0]]
        explicit_shapes[node.output[0]] = value
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")
        return

    raise RuntimeError("Should not reach here.")


def _infer_shape_of_transpose(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    attrs = get_onnx_attrs(node, initializers)
    perm = attrs["perm"]

    # There are two cases:
    # (1) Transpose an initializer
    # (2) Transpose a tensor value

    if node.input[0] in shapes:
        shape = [shapes[node.input[0]][i] for i in perm]
        shapes[node.output[0]] = shape
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")
    elif node.input[0] in initializers:
        value = [explicit_shapes[node.input[0]][i] for i in perm]
        explicit_shapes[node.output[0]] = value
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")
    else:
        raise RuntimeError("Should not reach here.")


def _infer_shape_of_unsqueeze(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    shapes: dict[str, list[int] | None],
    explicit_shapes: dict[str, Any],
):
    is_value = False
    if node.input[0] in shapes:
        data_shape = shapes[node.input[0]]
    elif node.input[0] in explicit_shapes:
        is_value = True
        data_shape = explicit_shapes[node.input[0]]
    else:
        raise RuntimeError("Should not reach here.")

    axes = onnx.numpy_helper.to_array(initializers[node.input[1]]).tolist()

    # If the data_shape is a single int, it means that the input is a scalar and we
    # want to expand it to a shape of [1]. This happends after a gather node.
    if type(data_shape) is int:
        assert axes == [0]
        value = [data_shape]
        explicit_shapes[node.output[0]] = value
        if _VERBOSE:
            print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")
        return

    def _infer_unsqueeze_shape(ori_shape_: list[int], axes_: list[int]) -> list[int]:
        new_shape = list(ori_shape_)
        for axis in sorted(axes_, reverse=True):
            new_shape.insert(axis, 1)
        return new_shape

    shape = _infer_unsqueeze_shape(data_shape, axes)
    if _VERBOSE:
        if is_value:
            explicit_shapes[node.output[0]] = shape
            print(f"Node {node.op_type:<20} {node.output[0]:<40} value={shape}")
        else:
            shapes[node.output[0]] = shape
            print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}")


INFER_SHAPE_FUNC_MAPPING = {
    "Gemm": _infer_shape_of_gemm,
    "Conv": _infer_shape_of_conv,
    "ConvTranspose": _infer_shape_of_convtranspose,
    "MatMul": _infer_shape_of_matmul,
    "Add": _infer_shape_of_binary_op,
    "Sub": _infer_shape_of_binary_op,
    "Mul": _infer_shape_of_binary_op,
    "Div": _infer_shape_of_binary_op,
    "Softmax": _infer_shape_of_act,
    "Relu": _infer_shape_of_act,
    "Sigmoid": _infer_shape_of_act,
    "Flatten": _infer_shape_of_flatten,
    "Shape": _infer_shape_of_shape,
    "Gather": _infer_shape_of_gather,
    "Slice": _infer_shape_of_slice,
    "Concat": _infer_shape_of_concat,
    "Reshape": _infer_shape_of_reshape,
    "Transpose": _infer_shape_of_transpose,
    "Unsqueeze": _infer_shape_of_unsqueeze,
    "ConstantOfShape": _infer_shape_of_constant_of_shape,
    "BatchNormalization": _infer_shape_of_batch_norm,
    "ReduceSum": _infer_shape_of_reduce,
    "ReduceMean": _infer_shape_of_reduce,
}


def infer_onnx_shape(
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    verbose: bool = False,
) -> dict[str, list[int]]:
    global _VERBOSE
    _VERBOSE = verbose
    """
    Two kinds of shapes in the model:
    1. Explicit shape: The shape is extracted by some nodes (e.g. Shape). The shape 
        itself is the data of the node.
    2. Shapes: The shape is inferred from the node and the node does not show 
       the shape. The shape is not the data of the node. 
    """
    data_shapes = {}
    explicit_shapes = {}

    if verbose:
        print("Inferring shape of input(s)...")
    _get_input_shape(input_nodes, data_shapes)

    if verbose:
        print("Inferring shape of output(s)...")
    _get_output_shape(output_nodes, data_shapes)

    if verbose:
        print("Inferring shape of initializer(s)...")
    _get_initializer_shape(initializers, data_shapes)

    if verbose:
        print("Inferring shape of node(s)...")
    for node in nodes:
        op_type = node.op_type
        if op_type == "Constant":
            raise RuntimeError(
                "There are constant nodes in the model, and you should convert them to "
                "initializers first by argument constant_to_initializer=True."
            )
        _infer_shape = INFER_SHAPE_FUNC_MAPPING[op_type]
        _infer_shape(node, initializers, data_shapes, explicit_shapes)

    data_shapes.update(explicit_shapes)

    return data_shapes
