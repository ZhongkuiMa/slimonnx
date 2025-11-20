__docformat__ = "restructuredtext"
__all__ = ["get_onnx_attrs"]

from typing import Any

from onnx import NodeProto, TensorProto

from .utils import EXTRACT_ATTR_MAP


def _scan_attrs(default_attrs: dict[str, Any], attrs) -> dict[str, Any]:
    for attr in attrs:
        extract = EXTRACT_ATTR_MAP.get(attr.type)
        if extract is None:
            raise NotImplementedError(f"{attr} with {attr.type} is not supported.")
        default_attrs[attr.name] = extract(attr)

    return default_attrs


def _check_pads(pads: tuple[int]):
    length = len(pads)
    dims = length // 2
    # Check the start pad and end pad are equal.
    for i in range(dims):
        if pads[i] != pads[i + dims]:
            raise ValueError(
                f"Pad node with unequal start and end {pads} is not supported."
            )


def _get_attrs_of_argmax(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ArgMax.html
    attrs = {
        "axis": 0,
        "keepdims": 1,  # True
        "select_last_index": 0,  # False
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["select_last_index"] != 0:
        raise ValueError(
            f"Argmax node with select_last_index={attrs['select_last_index']}"
            f"is not supported."
        )

    return attrs


def _get_attrs_of_avgpool(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__AveragePool.html
    attrs = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,  # False
        "count_include_pad": 0,  # False
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["kernel_shape"] is None:
        raise ValueError("AvgPool node requires kernel_shape attribute.")

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"AvgPool node with auto_pad={attrs['auto_pad']} is not supported."
        )

    if attrs["dilations"] is None:
        attrs["dilations"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["strides"] is None:
        attrs["strides"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["pads"] is None:
        attrs["pads"] = tuple([0] * len(attrs["kernel_shape"]) * 2)

    _check_pads(attrs["pads"])

    return attrs


def _get_attrs_of_batchnorm(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    attrs = {
        "epsilon": 1e-5,
        "momentum": 0.9,
        "training_mode": 0,  # False
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["training_mode"] != 0:
        raise ValueError(
            f"Batchnorm node with training_mode={attrs['training_mode']} is not supported."
        )

    if len(node.output) > 1:
        raise ValueError(
            f"Batchnorm node with {len(node.output)} outputs is not supported."
        )

    return attrs


def _get_attrs_of_cast(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Cast.html
    attrs = {"saturate": 1, "to": None}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["to"] is None:
        raise ValueError("Cast node requires 'to' attribute.")

    if attrs["saturate"] != 1:
        raise ValueError(
            f"Cast node with saturate={attrs['saturate']} is not supported."
        )

    return attrs


def _get_attrs_of_concat(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Concat.html
    attrs = {"axis": None}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["axis"] is None:
        raise ValueError("Concat node requires 'axis' attribute.")

    return attrs


def _get_attrs_of_conv(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Conv.html
    attrs = {
        "auto_pad": "NOTSET",
        "dilations": None,
        "group": 1,
        "kernel_shape": None,
        "pads": None,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Conv node with auto_pad={attrs['auto_pad']} is not supported."
        )

    if attrs["kernel_shape"] is None:
        # Infer the shape from the weight tensor.
        weight = initializers[node.input[1]]
        shape = tuple(weight.dims)
        attrs["kernel_shape"] = shape[2:]
    if attrs["dilations"] is None:
        # Infer the dilations from the kernel shape.
        attrs["dilations"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["strides"] is None:
        # Infer the strides from the kernel shape.
        attrs["strides"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["pads"] is None:
        # Infer the pads from the kernel shape.
        attrs["pads"] = tuple([0] * len(attrs["kernel_shape"]) * 2)

    _check_pads(attrs["pads"])

    return attrs


def _get_attrs_of_constant(*args, **kwargs) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Constant.html
    raise RuntimeError(
        "Constant is not supported. You should convert it to an initializer."
    )


def _get_attrs_of_convtranspose(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    attrs = {
        "auto_pad": "NOTSET",
        "dilations": None,
        "group": 1,
        "kernel_shape": None,
        "output_padding": None,
        "output_shape": None,
        "pads": None,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["group"] != 1:
        raise ValueError(
            f"ConvTranspose node with group={attrs['group']} is not supported."
        )
    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"ConvTranspose node with auto_pad={attrs['auto_pad']} is not supported."
        )
    if attrs["kernel_shape"] is None:
        # Infer the shape from the weight tensor.
        weight = initializers[node.input[1]]
        shape = tuple(weight.dims)
        attrs["kernel_shape"] = shape[2:]
    if attrs["dilations"] is None:
        # Infer the dilations from the kernel shape.
        attrs["dilations"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["strides"] is None:
        # Infer the strides from the kernel shape.
        attrs["strides"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["pads"] is None:
        # Infer the pads from the kernel shape.
        attrs["pads"] = tuple([0] * len(attrs["kernel_shape"]) * 2)
    if attrs["output_padding"] is None:
        # Infer the output padding from the kernel shape.
        attrs["output_padding"] = tuple([0] * len(attrs["kernel_shape"]))

    _check_pads(attrs["pads"])

    return attrs


def _get_attrs_of_constantofshape(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html
    attrs = {"value": None}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["value"] is None:
        raise ValueError("ConstantOfShape node requires 'value' attribute.")

    return attrs


def _get_attrs_of_elu(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Elu.html
    attrs = {"alpha": 1.0}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_flatten(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Flatten.html
    attrs = {"axis": 1}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gather(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Gather.html
    attrs = {"axis": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gelu(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Gelu.html
    attrs = {"approximate": "none"}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gemm(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Gemm.html
    attrs = {
        "alpha": 1.0,
        "beta": 1.0,
        "transA": 0,
        "transB": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_leakyrelu(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
    attrs = {"alpha": 0.01}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_maxpool(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__MaxPool.html
    attrs = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,  # False
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "storage_order": 0,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"MaxPool node with auto_pad={attrs['auto_pad']} is not supported."
        )
    if attrs["storage_order"] != 0:
        raise ValueError(
            f"MaxPool node with storage_order={attrs['storage_order']} is not supported."
        )
    if attrs["dilations"] is None:
        attrs["dilations"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["strides"] is None:
        attrs["strides"] = tuple([1] * len(attrs["kernel_shape"]))
    if attrs["pads"] is None:
        attrs["pads"] = tuple([0] * len(attrs["kernel_shape"]) * 2)

    _check_pads(attrs["pads"])

    if len(node.output) > 1:
        raise ValueError(
            f"MaxPool node with {len(node.output)} outputs is not supported."
        )

    return attrs


def _get_attrs_of_pad(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Pad.html
    attrs = {"mode": "constant"}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_reducemean(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ReduceMean.html
    attrs = {
        "keepdims": 1,  # True
        "noop_with_empty_axes": 0,  # False
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["noop_with_empty_axes"] != 0:
        raise ValueError(
            f"ReduceMean node with noop_with_empty_axes={attrs['noop_with_empty_axes']} "
            f"is not supported."
        )

    return attrs


def _get_attrs_of_reducesum(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ReduceSum.html
    attrs = {
        "keepdims": 1,  # True
        "noop_with_empty_axes": 0,  # False
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["noop_with_empty_axes"] != 0:
        raise ValueError(
            f"ReduceSum node with noop_with_empty_axes={attrs['noop_with_empty_axes']} "
            f"is not supported."
        )

    return attrs


def _get_attrs_of_reshape(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Reshape.html
    attrs = {"allowzero": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["allowzero"] != 0:
        raise ValueError(
            f"Reshape node with allowzero={attrs['allowzero']} is not supported."
        )

    return attrs


def _get_attrs_of_resize(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Resize.html
    attrs = {
        "antialias": 0,
        "axes": None,
        "coordinate_transformation_mode": "half_pixel",
        "cubic_coeff_a": -0.75,
        "exclude_outside": 0,
        "extrapolation_value": 0.0,
        "keep_aspect_ratio_policy": "stretch",
        "mode": "nearest",
        "nearest_mode": "round_prefer_floor",
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_scatter(*args, **kwargs) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Scatter.html
    # This will be replaced by ScatterElements
    return _get_attrs_of_scatter(*args, **kwargs)


def _get_attrs_of_scatterelement(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ScatterElements.html
    attrs = {"axis": 0, "reduction": "none"}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["reduction"] != "none":
        raise ValueError(
            f"Scatter node with reduction={attrs['reduction']} is not supported."
        )

    return attrs


def _get_attrs_of_scatternd(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__ScatterND.html
    attrs = {"reduction": "none"}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["reduction"] != "none":
        raise ValueError(
            f"ScatterND node with reduction={attrs['reduction']} is not supported."
        )

    return attrs


def _get_attrs_of_shape(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Shape.html
    attrs = {"end": -1, "start": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["end"] != -1:
        raise ValueError(f"Shape node with end={attrs['end']} is not supported.")
    if attrs["start"] != 0:
        raise ValueError(f"Shape node with start={attrs['start']} is not supported.")

    return attrs


def _get_attrs_of_softmax(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Softmax.html
    attrs = {"axis": -1}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_split(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Split.html
    attrs = {"axis": 0, "num_outputs": None}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_transpose(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Transpose.html
    attrs = {"perm": None}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["perm"] is None:
        raise ValueError("Transpose node requires 'perm' attribute.")

    return attrs


def _get_attrs_of_unsqueeze(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    attrs = {"axes": None}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_upsample(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Upsample.html
    attrs = {"mode": "nearest"}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


_EXTRACT_ATTRS_MAP = {
    "ArgMax": _get_attrs_of_argmax,
    "AveragePool": _get_attrs_of_avgpool,
    "BatchNormalization": _get_attrs_of_batchnorm,
    "Cast": _get_attrs_of_cast,
    "Concat": _get_attrs_of_concat,
    "Conv": _get_attrs_of_conv,
    "Constant": _get_attrs_of_constant,
    "ConvTranspose": _get_attrs_of_convtranspose,
    "ConstantOfShape": _get_attrs_of_constantofshape,
    "Elu": _get_attrs_of_elu,
    "Flatten": _get_attrs_of_flatten,
    "Gather": _get_attrs_of_gather,
    "Gelu": _get_attrs_of_gelu,
    "Gemm": _get_attrs_of_gemm,
    "LeakyRelu": _get_attrs_of_leakyrelu,
    "MaxPool": _get_attrs_of_maxpool,
    "Pad": _get_attrs_of_pad,
    "ReduceMean": _get_attrs_of_reducemean,
    "ReduceSum": _get_attrs_of_reducesum,
    "Reshape": _get_attrs_of_reshape,
    "Resize": _get_attrs_of_resize,
    "Shape": _get_attrs_of_shape,
    "Scatter": _get_attrs_of_scatter,
    "ScatterElements": _get_attrs_of_scatterelement,
    "ScatterND": _get_attrs_of_scatternd,
    "Softmax": _get_attrs_of_softmax,
    "Split": _get_attrs_of_split,
    "Transpose": _get_attrs_of_transpose,
    "Unsqueeze": _get_attrs_of_unsqueeze,
    "Upsample": _get_attrs_of_upsample,
}


def get_onnx_attrs(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Get attributes of the ONNX node.

    :param node: The ONNX node
    :param initializers: The initializers of the ONNX model
    :return: Dictionary of attributes with key as the name of the attribute
    """
    _get_attrs = _EXTRACT_ATTRS_MAP.get(node.op_type)
    if _get_attrs is None:
        raise NotImplementedError(f"Unsupported operator: {node.op_type}")
    return _get_attrs(node, initializers)
