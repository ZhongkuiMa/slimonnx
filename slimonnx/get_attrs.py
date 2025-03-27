__docformat__ = "restructuredtext"
__all__ = ["get_onnx_node_attrs"]

import onnx


def _get_attrs_of_conv(node: onnx.NodeProto) -> dict:
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
    for attr in node.attribute:
        if attr.name == "auto_pad":
            attrs["auto_pad"] = attr.s.decode("utf-8")
        elif attr.name == "dilations":
            attrs["dilations"] = list(attr.ints)
        elif attr.name == "group":
            attrs["group"] = attr.i
        elif attr.name == "kernel_shape":
            attrs["kernel_shape"] = list(attr.ints)
        elif attr.name == "output_padding":
            attrs["output_padding"] = list(attr.ints)
        elif attr.name == "output_shape":
            attrs["output_shape"] = list(attr.ints)
        elif attr.name == "pads":
            attrs["pads"] = list(attr.ints)
        elif attr.name == "strides":
            attrs["strides"] = list(attr.ints)

    if attrs["group"] != 1:
        raise ValueError(f"Only support group=1 but group={attrs['group']}.")
    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )

    if attrs["output_padding"] is None:  # Default value.
        attrs["output_padding"] = [0] * len(attrs["strides"])

    return attrs


def _get_attrs_of_pool(node: onnx.NodeProto) -> dict:
    attrs = {
        "auto_pad": "NOTSET",
        "ceil_mode": 0,
        "count_include_pad": 0,
        "dilations": None,
        "kernel_shape": None,
        "pads": None,
        "storage_order": 0,
        "strides": None,
    }
    for attr in node.attribute:
        if attr.name == "auto_pad":
            attrs["auto_pad"] = attr.s.decode("utf-8")
        elif attr.name == "ceil_mode":
            attrs["ceil_mode"] = attr.i
        elif attr.name == "count_include_pad":
            attrs["count_include_pad"] = attr.i
        elif attr.name == "dilations":
            attrs["dilations"] = list(attr.ints)
        elif attr.name == "kernel_shape":
            attrs["kernel_shape"] = list(attr.ints)
        elif attr.name == "pads":
            attrs["pads"] = list(attr.ints)
        elif attr.name == "storage_order":
            attrs["storage_order"] = attr.i
        elif attr.name == "strides":
            attrs["strides"] = list(attr.ints)

    if attrs["auto_pad"] != "NOTSET":
        raise ValueError(
            f"Only support auto_pad=NOTSET but auto_pad={attrs['auto_pad']}."
        )
    if attrs["ceil_mode"] != 0:
        raise ValueError(
            f"Only support ceil_mode=0 but ceil_mode={attrs['ceil_mode']}."
        )
    if attrs["count_include_pad"] != 0:
        raise ValueError(
            f"Only support count_include_pad=0 "
            f"but count_include_pad={attrs['count_include_pad']}."
        )
    if attrs["storage_order"] != 0:
        raise ValueError(
            f"Only support storage_order=0 but storage_order={attrs['storage_order']}."
        )

    return attrs


def _get_attrs_of_gemm(node: onnx.NodeProto) -> dict:
    attrs = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
    for attr in node.attribute:
        if attr.name == "alpha":
            attrs["alpha"] = attr.f
        elif attr.name == "beta":
            attrs["beta"] = attr.f
        elif attr.name == "transA":
            attrs["transA"] = attr.i
        elif attr.name == "transB":
            attrs["transB"] = attr.i

    return attrs


def _get_attrs_of_leakyrelu(node: onnx.NodeProto) -> dict:
    attrs = {"alpha": 0.01}
    for attr in node.attribute:
        if attr.name == "alpha":
            attrs["alpha"] = attr.f

    if attrs["alpha"] != 0.01:
        raise ValueError(f"Only support alpha=0.01 but alpha={attrs['alpha']}.")

    return attrs


def _get_attrs_of_flatten_gether_concat_softmax(node: onnx.NodeProto) -> dict:
    attrs = {"axis": 0}
    for attr in node.attribute:
        if attr.name == "axis":
            attrs["axis"] = attr.i

    return attrs


def _get_attrs_of_reshape(node: onnx.NodeProto) -> dict:
    attrs = {"allowzero": 0}
    for attr in node.attribute:
        if attr.name == "allowzero":
            attrs["allowzero"] = attr.i

    if attrs["allowzero"] != 0:
        raise ValueError(f"Only support allowzero=0 but allowzero={attrs['allowzero']}")

    return attrs


def _get_attrs_of_pad(node: onnx.NodeProto) -> dict:
    attrs = {"mode": "constant", "pads": None, "value": 0.0}
    for attr in node.attribute:
        if attr.name == "mode":
            attrs["mode"] = attr.s.decode("utf-8")
        elif attr.name == "pads":
            attrs["pads"] = list(attr.ints)
        elif attr.name == "value":
            attrs["value"] = attr.f

    return attrs


def _get_attrs_of_unsqueeze(node: onnx.NodeProto) -> dict:
    attrs = {"axes": None}
    for attr in node.attribute:
        if attr.name == "axes":
            attrs["axes"] = list(attr.ints)

    return attrs


def _get_attrs_of_transpose(node: onnx.NodeProto) -> dict:
    attrs = {"perm": None}
    for attr in node.attribute:
        if attr.name == "perm":
            attrs["perm"] = list(attr.ints)

    return attrs


def _get_attrs_of_split(node: onnx.NodeProto) -> dict:
    attrs = {"axis": 0, "num_outputs": None}
    for attr in node.attribute:
        if attr.name == "axis":
            attrs["axis"] = attr.i
        elif attr.name == "num_outputs":
            attrs["num_outputs"] = attr.i

    return attrs


def _get_attrs_of_reduce(node: onnx.NodeProto) -> dict:
    attrs = {"axes": None, "keepdims": 1, "noop_with_empty_axes": 0}
    for attr in node.attribute:
        if attr.name == "axes":
            attrs["axes"] = list(attr.ints)
        elif attr.name == "keepdims":
            attrs["keepdims"] = attr.i
        elif attr.name == "noop_with_empty_axes":
            attrs["noop_with_empty_axes"] = attr.i

    if attrs["noop_with_empty_axes"] != 0:
        raise ValueError(
            f"Only support noop_with_empty_axes=0 "
            f"but noop_with_empty_axes={attrs['noop_with_empty_axes']}"
        )

    return attrs


def _get_attrs_of_batchnorm(node: onnx.NodeProto) -> dict:
    attrs = {"epsilon": 1e-5}
    for attr in node.attribute:
        if attr.name == "epsilon":
            attrs["epsilon"] = attr.f

    return attrs


def _get_attrs_of_cast(node: onnx.NodeProto) -> dict:
    attrs = {"to": None, "saturate": 1}
    for attr in node.attribute:
        if attr.name == "to":
            attrs["to"] = attr.i
        elif attr.name == "saturate":
            attrs["saturate"] = attr.i

    if attrs["saturate"] != 1:
        raise ValueError(f"Only support saturate=1 but saturate={attrs['saturate']}")

    return attrs


def _get_attrs_of_scatter(node: onnx.NodeProto) -> dict:
    attrs = {"reduction": "none"}
    for attr in node.attribute:
        if attr.name == "reduction":
            attrs["reduction"] = attr.s.decode("utf-8")

    if attrs["reduction"] != "none":
        raise ValueError(
            f"Only support reduction=none but reduction={attrs['reduction']}"
        )

    return attrs


def _get_attrs_of_upsample(node: onnx.NodeProto) -> dict:
    attrs = {"mode": "nearest"}
    for attr in node.attribute:
        if attr.name == "mode":
            attrs["mode"] = attr.s.decode("utf-8")

    return attrs


def _get_attrs_of_argmax(node: onnx.NodeProto) -> dict:
    attrs = {"axis": 0, "keepdims": 1, "select_last_index": 0}
    for attr in node.attribute:
        if attr.name == "axis":
            attrs["axis"] = attr.i
        elif attr.name == "keepdims":
            attrs["keepdims"] = attr.i
        elif attr.name == "select_last_index":
            attrs["select_last_index"] = attr.i

    if attrs["select_last_index"] != 0:
        raise ValueError(
            f"Only support select_last_index=0 "
            f"but select_last_index={attrs['select_last_index']}"
        )

    return attrs


def _get_attrs_of_resize(node: onnx.NodeProto) -> dict:
    attrs = {
        "antialias": 0,
        "axes": None,
        "coordinate_transformation_mode": "half_pixel",
        "cubic_coeff_a": -0.75,
        "exclude_outside": 0,
        "extrapolation_value": 0.0,
        "keep_aspect_ratio_policy": "keep",
        "mode": "nearest",
        "nearest_mode": "round_prefer_floor",
    }
    for attr in node.attribute:
        if attr.name == "antialias":
            attrs["antialias"] = attr.i
        elif attr.name == "axes":
            attrs["axes"] = list(attr.ints)
        elif attr.name == "coordinate_transformation_mode":
            attrs["coordinate_transformation_mode"] = attr.s.decode("utf-8")
        elif attr.name == "cubic_coeff_a":
            attrs["cubic_coeff_a"] = attr.f
        elif attr.name == "exclude_outside":
            attrs["exclude_outside"] = attr.i
        elif attr.name == "extrapolation_value":
            attrs["extrapolation_value"] = attr.f
        elif attr.name == "keep_aspect_ratio_policy":
            attrs["keep_aspect_ratio_policy"] = attr.s.decode("utf-8")
        elif attr.name == "mode":
            attrs["mode"] = attr.s.decode("utf-8")
        elif attr.name == "nearest_mode":
            attrs["nearest_mode"] = attr.s.decode("utf-8")

    return attrs


GET_ATTRS_FUNC_MAPPING = {
    "Conv": _get_attrs_of_conv,
    "ConvTranspose": _get_attrs_of_conv,
    "Gemm": _get_attrs_of_gemm,
    "LeakyRelu": _get_attrs_of_leakyrelu,
    "Flatten": _get_attrs_of_flatten_gether_concat_softmax,
    "Gather": _get_attrs_of_flatten_gether_concat_softmax,
    "Concat": _get_attrs_of_flatten_gether_concat_softmax,
    "Softmax": _get_attrs_of_flatten_gether_concat_softmax,
    "Reshape": _get_attrs_of_reshape,
    "Pad": _get_attrs_of_pad,
    "Unsqueeze": _get_attrs_of_unsqueeze,
    "Transpose": _get_attrs_of_transpose,
    "Split": _get_attrs_of_split,
    "ReduceMean": _get_attrs_of_reduce,
    "ReduceSum": _get_attrs_of_reduce,
    "BatchNormalization": _get_attrs_of_batchnorm,
    "Cast": _get_attrs_of_cast,
    "ScatterND": _get_attrs_of_scatter,
    "Upsample": _get_attrs_of_upsample,
    "ArgMax": _get_attrs_of_argmax,
    "Resize": _get_attrs_of_resize,
}


def get_onnx_node_attrs(node: onnx.NodeProto):
    """
    Get attributes of the ONNX node.

    :param node: The ONNX node.

    :return: A dictionary of attributes with key is the name of the attribute.
    """

    _get_attrs = GET_ATTRS_FUNC_MAPPING[node.op_type]
    attrs = _get_attrs(node)

    return attrs
