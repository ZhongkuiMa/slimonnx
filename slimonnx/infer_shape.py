__docformat__ = "restructuredtext"
__all__ = ["infer_onnx_shape"]

import math
from math import ceil, floor
from typing import Any

import numpy as np
import onnx

from slimonnx.get_attrs import get_onnx_node_attrs
from slimonnx.utils import *

_VERBOSE = False


def _get_input_shape(model: onnx.ModelProto, all_shapes: dict[str, list[int]]):
    for input_node in model.graph.input:
        shape = [x.dim_value for x in input_node.type.tensor_type.shape.dim[1:]]
        shape = [1] + shape
        shape = [int(x) for x in shape]
        all_shapes[input_node.name] = shape
        if _VERBOSE:
            print(f"Input {input_node.name:<50} Shape={shape}")


def _get_output_shape(model: onnx.ModelProto, all_shapes: dict[str, list[int]]):
    for output_node in model.graph.output:
        shape = [x.dim_value for x in output_node.type.tensor_type.shape.dim[1:]]
        shape = [1] + shape
        shape = [int(x) for x in shape]
        all_shapes[output_node.name] = shape
        if _VERBOSE:
            print(f"Output {output_node.name:<50} Shape={shape}")


def _get_initializer_shape(model: onnx.ModelProto, all_shapes: dict[str, list[int]]):
    for initializer in model.graph.initializer:
        shape = [int(x) for x in initializer.dims]
        all_shapes[initializer.name] = shape
        if _VERBOSE:
            print(f"Initializer {initializer.name:<50} Shape={shape}")


def _infer_shape_of_gemm(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    attrs = get_onnx_node_attrs(node)
    transA = attrs["transA"]
    transB = attrs["transB"]
    shape1 = all_shapes[node.input[0]]
    shape2 = all_shapes[node.input[1]]

    if transA:
        shape1 = shape1[:-2] + [shape1[-1], shape1[-2]]
    if transB:
        shape2 = shape2[:-2] + [shape2[-1], shape2[-2]]

    shape = shape1[:-1] + shape2[-1:]

    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_matmul(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape1 = all_shapes[node.input[0]]
    shape2 = all_shapes[node.input[1]]

    if not (len(shape1) >= 2 and len(shape2) >= 2 and shape1[-1] == shape2[-2]):
        raise NotImplementedError(
            f"Not supported {node.op_type:<20} with shape {shape1} and {shape2}"
        )

    shape = [*shape1[:-1], shape2[-1]]
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_binary_op(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape1 = all_shapes[node.input[0]]
    shape2 = all_shapes[node.input[1]]

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

    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_conv(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    weight = initializers[node.input[1]]
    weight_shape = list(weight.dims)
    attrs = get_onnx_node_attrs(node)
    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    input_shape = all_shapes[node.input[0]]
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

    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_conv_transpose(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    weight = initializers[node.input[1]]
    weight_shape = list(weight.dims)
    attrs = get_onnx_node_attrs(node)

    kernel_shape = attrs["kernel_shape"]
    dilations = attrs["dilations"]
    output_padding = attrs["output_padding"]
    pads = attrs["pads"]
    strides = attrs["strides"]
    input_shape = all_shapes[node.input[0]]
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

    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_flatten(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape = all_shapes[node.input[0]]
    axis = get_onnx_node_attrs(node)["axis"]
    shape = shape[:axis] + [-1]
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_act(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape = all_shapes[node.input[0]]
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_shape(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    value = all_shapes[node.input[0]]
    temp_values[node.output[0]] = value
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} value={value}")


def _infer_shape_of_gather(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    attrs = get_onnx_node_attrs(node)
    axis = attrs["axis"]
    indices = onnx.numpy_helper.to_array(initializers[node.input[1]])
    value = temp_values[node.input[0]]
    if type(value[axis]) == int:
        value = value[indices]
        shape = None
    else:
        raise NotImplementedError

    temp_values[node.output[0]] = value
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}, value={value}"
        )


def _infer_shape_of_slice(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    starts = initializers[node.input[1]]
    ends = initializers[node.input[2]]
    axes = initializers[node.input[3]]
    starts = onnx.numpy_helper.to_array(starts).astype(int).tolist()
    ends = onnx.numpy_helper.to_array(ends).astype(int).tolist()
    axes = onnx.numpy_helper.to_array(axes).astype(int).tolist()
    if len(node.input) == 5:
        steps = initializers[node.input[4]]
        steps = onnx.numpy_helper.to_array(steps)
        steps = [int(x) for x in steps]
    else:
        steps = [1] * len(axes)

    is_value = False
    value = all_shapes.get(node.input[0], None)
    if value is None:
        value = temp_values[node.input[0]]

    if type(value[0]) == int:
        value = value[starts[0] : ends[0] : steps[0]]
        shape = None
    else:
        value = None
        raise NotImplementedError

    temp_values[node.output[0]] = value
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}, value={value}"
        )


def _infer_shape_of_concat(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    attrs = get_onnx_node_attrs(node)
    axis = attrs["axis"]

    is_value = False
    shapes = []
    for input_i in node.input:
        shapes.append(temp_values.get(input_i, None))

    if any([shape is not None for shape in shapes]):
        is_value = True

    for i, (input_i, shape_i) in enumerate(zip(node.input, shapes)):
        if shape_i is None:
            initializer = initializers.get(input_i, None)
            if initializer is not None:
                data = onnx.numpy_helper.to_array(initializer)
                if np.issubdtype(data.dtype, np.integer):
                    shapes[i] = data.tolist()
                else:
                    shapes[i] = list(data.shape)
            else:
                shapes[i] = all_shapes.get(input_i, None)

    if is_value:
        if axis == 0:
            shape = []
            for shape_i in shapes:
                shape.extend(shape_i)

        else:
            raise NotImplementedError
    else:
        shape = shapes[0].copy()
        shape[axis] = 0
        for shape_i in shapes:
            shape[axis] += shape_i[axis]

    value = None
    shape, value = (value, shape) if is_value else (shape, value)
    temp_values[node.output[0]] = value
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}, value={value}"
        )


def _infer_shape_of_reshape(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    data_shape = all_shapes[node.input[0]]
    shape = initializers.get(node.input[1], None)
    if shape is not None:
        shape = onnx.numpy_helper.to_array(shape).astype(int).tolist()
    else:
        shape = temp_values[node.input[1]]

    inferred_shape = math.prod(data_shape)
    inferred_idx = -1
    for idx, data_shape_i in enumerate(shape):
        if data_shape_i == -1:
            inferred_idx = idx
            continue
        inferred_shape = inferred_shape / data_shape_i
    if inferred_idx != -1:
        shape[inferred_idx] = int(inferred_shape)

    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_transpose(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    attrs = get_onnx_node_attrs(node)
    perm = attrs["perm"]

    is_value = False
    shape = all_shapes.get(node.input[0], None)
    if shape is None:
        is_value = True
        shape = initializers[node.input[0]]

    value = None
    shape = [shape[i] for i in perm]
    shape, value = (value, shape) if is_value else (shape, value)
    temp_values[node.output[0]] = value
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}, value={value}"
        )


def _infer_shape_of_unsqueeze(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    is_value = False
    shape = all_shapes.get(node.input[0], None)
    if shape is None:
        is_value = True
        shape = temp_values[node.input[0]]

    if len(node.input) == 2:
        # New version of ONNX
        axes = onnx.numpy_helper.to_array(initializers[node.input[1]])
    else:
        axes = get_onnx_node_attrs(node)["axes"]

    for axis in axes:
        if type(shape) == int:
            shape = [shape]
        else:
            shape.insert(axis, 1)

    value = None
    shape, value = (value, shape) if is_value else (shape, value)
    temp_values[node.output[0]] = value
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape}, value={value}"
        )


def _infer_shape_of_constant_of_shape(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape = temp_values[node.input[0]]
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


def _infer_shape_of_batch_norm(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    all_shapes[node.output[0]] = all_shapes[node.input[0]]
    if _VERBOSE:
        print(
            f"Node {node.op_type:<20} {node.output[0]:<40}shape={all_shapes[node.output[0]]}, value=None"
        )


def _infer_shape_of_reduce(
    node: onnx.NodeProto,
    all_shapes: dict[str, list[int] | None],
    initializers: dict[str, onnx.TensorProto],
    temp_values: dict[str, Any],
):
    shape = all_shapes[node.input[0]].copy()
    axes = get_onnx_node_attrs(node)["axes"]
    keepdims = get_onnx_node_attrs(node)["keepdims"]
    for axis in axes:
        shape[axis] = 1 if keepdims else 0
    # Remove all 0 in shape
    shape = [x for x in shape if x != 0]
    all_shapes[node.output[0]] = shape
    if _VERBOSE:
        print(f"Node {node.op_type:<20} {node.output[0]:<40} shape={shape} value=None")


INFER_SHAPE_FUNC_MAPPING = {
    "Gemm": _infer_shape_of_gemm,
    "Conv": _infer_shape_of_conv,
    "ConvTranspose": _infer_shape_of_conv_transpose,
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
    model: onnx.ModelProto,
    verbose: bool = False,
) -> dict[str, list[int]]:
    """
    Infer the shape of all nodes in the model.

    :param model: The ONNX model.
    :param verbose: Whether to print the shape of each node.

    :return: A dictionary with the shape of each node.
    """
    global _VERBOSE
    _VERBOSE = verbose

    if verbose:
        print(
            "Inferring shape for model...",
            "Model info:",
            f"model_version={model.model_version}",
            f"ir_version={model.ir_version}",
            f"opset_import={model.opset_import}",
            f"producer_name={model.producer_name}",
            f"producer_version={model.producer_version}",
            f"doc_string={model.doc_string}",
            f"DESCRIPTOR={model.DESCRIPTOR}",
            sep="\n",
        )

    all_shapes = {}
    if verbose:
        print("Inferring shape of input(s)...")
    _get_input_shape(model, all_shapes)

    if verbose:
        print("Inferring shape of output(s)...")
    _get_output_shape(model, all_shapes)

    if verbose:
        print("Inferring shape of initializer(s)...")
    _get_initializer_shape(model, all_shapes)

    if verbose:
        print("Getting initializer(s)...")
    initializers = get_initializers(model)

    temp_values = {}

    if verbose:
        print("Inferring shape of node(s)...")
    for node in model.graph.node:
        op_type = node.op_type
        _infer_shape = INFER_SHAPE_FUNC_MAPPING[op_type]
        _infer_shape(node, all_shapes, initializers, temp_values)

    return all_shapes
