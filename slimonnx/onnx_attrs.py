__docformat__ = "restructuredtext"
__all__ = ["get_attrs_of_onnx_node"]

import warnings
from typing import Any

import onnx

from .utils import EXTRACT_ATTR_MAP


def _scan_attrs(default_attrs: dict[str, Any], attrs) -> dict[str, Any]:
    for attr in attrs:
        extract = EXTRACT_ATTR_MAP.get(attr.type)
        if extract is None:
            raise NotImplementedError(f"Invalid attribute type: {attr}")
        default_attrs[attr.name] = extract(attr)

    return default_attrs


def _get_attrs_of_add(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Add.html
    return {}


def _get_attrs_of_argmax(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__ArgMax.html
    attrs = {
        "axis": 0,
        "keepdims": 1,
        "select_last_index": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["select_last_index"] != 0:
        raise ValueError(
            f"Only support select_last_index=0 "
            f"but select_last_index={attrs['select_last_index']}"
        )

    return attrs


def _get_attrs_of_averagepool(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__AveragePool.html
    attrs = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "count_include_pad": 0,
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )
    if attrs["dilations"] is None:
        attrs["dilations"] = [1] * len(attrs["kernel_shape"])
    if attrs["strides"] is None:
        attrs["strides"] = [1] * len(attrs["kernel_shape"])
    if attrs["pads"] is None:
        attrs["pads"] = [0] * len(attrs["kernel_shape"]) * 2

    return attrs


def _get_attrs_of_batchnormalization(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
    attrs = {
        "epsilon": 1e-5,
        "momentum": 0.9,
        "training_mode": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if abs(attrs["momentum"] - 0.9) > 1e-5:
        raise ValueError(f"Only support momentum=0.9 but momentum={attrs['momentum']}")
    if attrs["training_mode"] != 0:
        raise ValueError(
            f"Only support training_mode=0 "
            f"but training_mode={attrs['training_mode']}"
        )
    return attrs


def _get_attrs_of_cast(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Cast.html
    attrs = {
        "saturate": 1,
        "to": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["saturate"] != 1:
        raise ValueError(f"Only support saturate=1 but saturate={attrs['saturate']}")

    return attrs


def _get_attrs_of_concat(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Concat.html
    attrs = {"axis": None}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_conv(node: onnx.NodeProto) -> dict[str, Any]:
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

    if attrs["group"] != 1:
        raise ValueError(f"Only support group=1 but group={attrs['group']}.")
    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )
    if attrs["dilations"] is None:
        attrs["dilations"] = [1] * len(attrs["kernel_shape"])
    if attrs["strides"] is None:
        attrs["strides"] = [1] * len(attrs["kernel_shape"])
    if attrs["pads"] is None:
        attrs["pads"] = [0] * len(attrs["kernel_shape"]) * 2

    return attrs


def _get_attrs_of_convtranspose(node: onnx.NodeProto) -> dict[str, Any]:
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
        raise ValueError(f"Only support group=1 but group={attrs['group']}.")
    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )
    if attrs["dilations"] is None:
        attrs["dilations"] = [1] * len(attrs["kernel_shape"])
    if attrs["strides"] is None:
        attrs["strides"] = [1] * len(attrs["kernel_shape"])
    if attrs["pads"] is None:
        attrs["pads"] = [0] * len(attrs["kernel_shape"]) * 2
    if attrs["output_padding"] is None:
        attrs["output_padding"] = [0] * len(attrs["kernel_shape"])

    return attrs


def _get_attrs_of_elu(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Elu.html
    attrs = {"alpha": 1.0}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_flatten(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Flatten.html
    attrs = {"axis": 1}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gather(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Gather.html
    attrs = {"axis": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gelu(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Gelu.html
    attrs = {"approximate": "none"}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_gemm(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Gemm.html
    attrs = {
        "alpha": 1.0,
        "beta": 1.0,
        "transA": 0,
        "transB": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_leakyrelu(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
    attrs = {"alpha": 0.01}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["alpha"] != 0.01:
        raise ValueError(f"Only support alpha=0.01 but alpha={attrs['alpha']}.")

    return attrs


def _get_attrs_of_maxpool(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__MaxPool.html
    attrs = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "storage_order": 0,
        "strides": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )
    if attrs["storage_order"] != 0:
        raise ValueError(
            f"Only support storage_order=0 but storage_order={attrs['storage_order']}."
        )
    if attrs["dilations"] is None:
        attrs["dilations"] = [1] * len(attrs["kernel_shape"])
    if attrs["strides"] is None:
        attrs["strides"] = [1] * len(attrs["kernel_shape"])
    if attrs["pads"] is None:
        attrs["pads"] = [0] * len(attrs["kernel_shape"]) * 2

    return attrs


def _get_attrs_of_pad(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Pad.html
    attrs = {
        "mode": "constant",
        "pads": None,  # TODO: This attr only exists in old version, try to remove it.
        "value": 0.0,  # TODO: This attr only exists in old version, try to remove it.
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_reducemean(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__ReduceMean.html
    attrs = {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["noop_with_empty_axes"] != 0:
        raise ValueError(
            f"Only support noop_with_empty_axes=0 "
            f"but noop_with_empty_axes={attrs['noop_with_empty_axes']}"
        )

    return attrs


def _get_attrs_of_relu(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Relu.html
    return {}


def _get_attrs_of_reducesum(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__ReduceSum.html
    attrs = {
        "keepdims": 1,
        "noop_with_empty_axes": 0,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["noop_with_empty_axes"] != 0:
        raise ValueError(
            f"Only support noop_with_empty_axes=0 "
            f"but noop_with_empty_axes={attrs['noop_with_empty_axes']}"
        )

    return attrs


def _get_attrs_of_reshape(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Reshape.html
    attrs = {"allowzero": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["allowzero"] != 0:
        raise ValueError(f"Only support allowzero=0 but allowzero={attrs['allowzero']}")

    return attrs


def _get_attrs_of_resize(node: onnx.NodeProto) -> dict:
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

    warnings.warn("This operator is experimental and has not been tested.")

    return attrs


def _get_attrs_of_scatter(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Scatter.html
    attrs = {"axis": 0}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_scatternd(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__ScatterND.html
    attrs = {"reduction": "none"}
    attrs = _scan_attrs(attrs, node.attribute)

    if attrs["reduction"] != "none":
        raise ValueError(
            f"Only support reduction=none but reduction={attrs['reduction']}"
        )

    return attrs


def _get_attrs_of_sigmoid(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Sigmoid.html
    return {}


def _get_attrs_of_softmax(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Softmax.html
    attrs = {"axis": -1}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_split(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Split.html
    attrs = {
        "axis": 0,
        "num_outputs": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_tanh(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Tanh.html
    return {}


def _get_attrs_of_transpose(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Transpose.html
    attrs = {"perm": None}
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_unsqueeze(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    attrs = {
        "axes": None,  # TODO: This attr only exists in old version, try to remove.
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


def _get_attrs_of_upsample(node: onnx.NodeProto) -> dict:
    # https://onnx.ai/onnx/operators/onnx__Upsample.html
    attrs = {
        "mode": "nearest",
        # TODO: This exists in old version, new version needs shape inference.
        "scales": None,
    }
    attrs = _scan_attrs(attrs, node.attribute)

    return attrs


_EXTRACT_ATTRS_MAPPING = {
    "Add": _get_attrs_of_add,
    "ArgMax": _get_attrs_of_argmax,
    "AveragePool": _get_attrs_of_averagepool,
    "BatchNormalization": _get_attrs_of_batchnormalization,
    "Cast": _get_attrs_of_cast,
    "Concat": _get_attrs_of_concat,
    "Conv": _get_attrs_of_conv,
    "ConvTranspose": _get_attrs_of_convtranspose,
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
    "Relu": _get_attrs_of_relu,
    "Reshape": _get_attrs_of_reshape,
    "Resize": _get_attrs_of_resize,
    "Sigmoid": _get_attrs_of_sigmoid,
    "Scatter": _get_attrs_of_scatter,
    "ScatterND": _get_attrs_of_scatternd,
    "Softmax": _get_attrs_of_softmax,
    "Split": _get_attrs_of_split,
    "Tanh": _get_attrs_of_tanh,
    "Transpose": _get_attrs_of_transpose,
    "Unsqueeze": _get_attrs_of_unsqueeze,
    "Upsample": _get_attrs_of_upsample,
}


def get_attrs_of_onnx_node(node: onnx.NodeProto):
    """
    Get attributes of the ONNX node.

    :param node: The ONNX node.

    :return: A dictionary of attributes with key is the name of the attribute.
    """

    _get_attrs = _EXTRACT_ATTRS_MAPPING.get(node.op_type)
    if _get_attrs is None:
        raise NotImplementedError(f"Unsupported operator: {node.op_type}")
    attrs = _get_attrs(node)

    return attrs
