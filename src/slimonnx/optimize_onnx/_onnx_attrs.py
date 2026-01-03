"""ONNX node attribute extraction and validation."""

__docformat__ = "restructuredtext"
__all__ = ["get_onnx_attrs"]

from collections.abc import Callable
from typing import Any

import onnx
from onnx import NodeProto, TensorProto

# Attribute type extractors
EXTRACT_ATTR_MAP: dict[int, Any] = {
    0: lambda x: None,  # UNDEFINED
    1: lambda x: x.f,  # FLOAT
    2: lambda x: x.i,  # INT
    3: lambda x: x.s.decode("utf-8"),  # STRING
    4: lambda x: onnx.numpy_helper.to_array(x.t),  # TENSOR
    5: lambda x: x.g,  # GRAPH
    6: lambda x: tuple(x.floats),  # FLOATS
    7: lambda x: tuple(x.ints),  # INTS
    8: lambda x: None,  # STRINGS
    9: lambda x: None,  # TENSORS
    10: lambda x: None,  # GRAPHS
    11: lambda x: None,  # SPARSE_TENSOR
}


def scan_attrs(default_attrs: dict[str, Any], attrs) -> dict[str, Any]:
    """
    Scan and extract ONNX node attributes.

    :param default_attrs: Default attribute values
    :param attrs: ONNX node attributes
    :return: Extracted attributes merged with defaults
    """
    result = default_attrs.copy()
    for attr in attrs:
        extract = EXTRACT_ATTR_MAP.get(attr.type)
        if extract is None:
            raise NotImplementedError(
                f"Attribute {attr.name} with type {attr.type} is not supported"
            )
        result[attr.name] = extract(attr)
    return result


def check_pads_symmetric(pads: tuple[int, ...]) -> None:
    """
    Verify that padding is symmetric.

    :param pads: Padding tuple
    """
    dims = len(pads) // 2
    for i in range(dims):
        if pads[i] != pads[i + dims]:
            raise ValueError(
                f"Asymmetric padding {pads} is not supported; start and end padding must be equal"
            )


def infer_kernel_defaults(attrs: dict[str, Any], kernel_shape: tuple[int, ...]) -> dict[str, Any]:
    """
    Infer default values for dilations, strides, and pads.

    :param attrs: Attribute dictionary
    :param kernel_shape: Kernel dimensions
    :return: Updated attributes with inferred defaults
    """
    kernel_dims = len(kernel_shape)
    if attrs.get("dilations") is None:
        attrs["dilations"] = tuple([1] * kernel_dims)
    if attrs.get("strides") is None:
        attrs["strides"] = tuple([1] * kernel_dims)
    if attrs.get("pads") is None:
        attrs["pads"] = tuple([0] * kernel_dims * 2)
    return attrs


def validate_auto_pad(auto_pad: str, op_name: str) -> None:
    """
    Validate auto_pad attribute is NOTSET.

    :param auto_pad: auto_pad value
    :param op_name: Operator name for error message
    """
    if auto_pad != "NOTSET":
        raise ValueError(f"{op_name} with auto_pad={auto_pad} is not supported")


def get_attrs_argmax(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract ArgMax operator attributes."""
    attrs = scan_attrs({"axis": 0, "keepdims": 1, "select_last_index": 0}, node.attribute)
    if attrs["select_last_index"] != 0:
        raise ValueError(
            f"ArgMax with select_last_index={attrs['select_last_index']} is not supported"
        )
    return attrs


def get_attrs_avgpool(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract AveragePool operator attributes."""
    attrs = scan_attrs(
        {
            "auto_pad": "NOTSET",
            "ceil_mode": 0,
            "count_include_pad": 0,
            "dilations": None,
            "kernel_shape": None,
            "pads": None,
            "strides": None,
        },
        node.attribute,
    )
    if attrs["kernel_shape"] is None:
        raise ValueError("AveragePool kernel_shape is required")
    validate_auto_pad(attrs["auto_pad"], "AveragePool")
    infer_kernel_defaults(attrs, attrs["kernel_shape"])
    check_pads_symmetric(attrs["pads"])
    return attrs


def get_attrs_batchnorm(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract BatchNormalization operator attributes."""
    attrs = scan_attrs({"epsilon": 1e-5, "momentum": 0.9, "training_mode": 0}, node.attribute)
    if attrs["training_mode"] != 0:
        raise ValueError(
            f"BatchNormalization with training_mode={attrs['training_mode']} is not supported"
        )
    if len(node.output) > 1:
        raise ValueError(f"BatchNormalization with {len(node.output)} outputs is not supported")
    return attrs


def get_attrs_cast(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Cast operator attributes."""
    attrs = scan_attrs({"saturate": 1, "to": None}, node.attribute)
    if attrs["to"] is None:
        raise ValueError("Cast 'to' attribute is required")
    if attrs["saturate"] != 1:
        raise ValueError(f"Cast with saturate={attrs['saturate']} is not supported")
    return attrs


def get_attrs_concat(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Concat operator attributes."""
    attrs = scan_attrs({"axis": None}, node.attribute)
    if attrs["axis"] is None:
        raise ValueError("Concat axis is required")
    return attrs


def get_attrs_conv(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Conv operator attributes."""
    attrs = scan_attrs(
        {
            "auto_pad": "NOTSET",
            "dilations": None,
            "group": 1,
            "kernel_shape": None,
            "pads": None,
            "strides": None,
        },
        node.attribute,
    )
    validate_auto_pad(attrs["auto_pad"], "Conv")

    if attrs["kernel_shape"] is None:
        weight = initializers[node.input[1]]
        attrs["kernel_shape"] = tuple(weight.dims[2:])

    infer_kernel_defaults(attrs, attrs["kernel_shape"])
    check_pads_symmetric(attrs["pads"])
    return attrs


def get_attrs_constant(*args, **kwargs) -> dict[str, Any]:
    """Extract Constant operator attributes."""
    raise RuntimeError("Constant nodes are not supported; convert them to initializers first")


def get_attrs_conv_transpose(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract ConvTranspose operator attributes."""
    attrs = scan_attrs(
        {
            "auto_pad": "NOTSET",
            "dilations": None,
            "group": 1,
            "kernel_shape": None,
            "output_padding": None,
            "output_shape": None,
            "pads": None,
            "strides": None,
        },
        node.attribute,
    )
    if attrs["group"] != 1:
        raise ValueError(f"ConvTranspose with group={attrs['group']} is not supported")
    validate_auto_pad(attrs["auto_pad"], "ConvTranspose")

    if attrs["kernel_shape"] is None:
        weight = initializers[node.input[1]]
        attrs["kernel_shape"] = tuple(weight.dims[2:])

    infer_kernel_defaults(attrs, attrs["kernel_shape"])
    if attrs["output_padding"] is None:
        attrs["output_padding"] = tuple([0] * len(attrs["kernel_shape"]))
    check_pads_symmetric(attrs["pads"])
    return attrs


def get_attrs_constantofshape(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract ConstantOfShape operator attributes."""
    attrs = scan_attrs({"value": None}, node.attribute)
    if attrs["value"] is None:
        raise ValueError("ConstantOfShape value is required")
    return attrs


def get_attrs_simple(defaults: dict[str, Any]) -> Callable:
    """
    Create a simple attribute extractor for operators with only defaults.

    :param defaults: Default attribute values
    :return: Attribute extraction function
    """

    def extractor(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
        return scan_attrs(defaults, node.attribute)

    return extractor


def get_attrs_maxpool(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract MaxPool operator attributes."""
    attrs = scan_attrs(
        {
            "auto_pad": "NOTSET",
            "ceil_mode": 0,
            "dilations": None,
            "kernel_shape": None,
            "pads": None,
            "storage_order": 0,
            "strides": None,
        },
        node.attribute,
    )
    if attrs["kernel_shape"] is None:
        raise ValueError("MaxPool kernel_shape is required")
    if attrs["storage_order"] != 0:
        raise ValueError(f"MaxPool with storage_order={attrs['storage_order']} is not supported")
    validate_auto_pad(attrs["auto_pad"], "MaxPool")
    infer_kernel_defaults(attrs, attrs["kernel_shape"])
    check_pads_symmetric(attrs["pads"])

    if len(node.output) > 1:
        raise ValueError(f"MaxPool with {len(node.output)} outputs is not supported")
    return attrs


def get_attrs_reduce(op_name: str) -> Callable:
    """
    Create attribute extractor for reduce operators.

    :param op_name: Operator name for error messages
    :return: Attribute extraction function
    """

    def extractor(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
        attrs = scan_attrs({"keepdims": 1, "noop_with_empty_axes": 0}, node.attribute)
        if attrs["noop_with_empty_axes"] != 0:
            raise ValueError(
                f"{op_name} with noop_with_empty_axes={attrs['noop_with_empty_axes']} "
                "is not supported"
            )
        return attrs

    return extractor


def get_attrs_reshape(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Reshape operator attributes."""
    attrs = scan_attrs({"allowzero": 0}, node.attribute)
    if attrs["allowzero"] != 0:
        raise ValueError(f"Reshape with allowzero={attrs['allowzero']} is not supported")
    return attrs


def get_attrs_resize(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Resize operator attributes."""
    return scan_attrs(
        {
            "antialias": 0,
            "axes": None,
            "coordinate_transformation_mode": "half_pixel",
            "cubic_coeff_a": -0.75,
            "exclude_outside": 0,
            "extrapolation_value": 0.0,
            "keep_aspect_ratio_policy": "stretch",
            "mode": "nearest",
            "nearest_mode": "round_prefer_floor",
        },
        node.attribute,
    )


def get_attrs_scatter(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Scatter operator attributes."""
    return get_attrs_scatterelement(node, initializers)


def get_attrs_scatterelement(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract ScatterElements operator attributes."""
    attrs = scan_attrs({"axis": 0, "reduction": "none"}, node.attribute)
    if attrs["reduction"] != "none":
        raise ValueError(f"ScatterElements with reduction={attrs['reduction']} is not supported")
    return attrs


def get_attrs_scatternd(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract ScatterND operator attributes."""
    attrs = scan_attrs({"reduction": "none"}, node.attribute)
    if attrs["reduction"] != "none":
        raise ValueError(f"ScatterND with reduction={attrs['reduction']} is not supported")
    return attrs


def get_attrs_shape(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Shape operator attributes."""
    attrs = scan_attrs({"end": -1, "start": 0}, node.attribute)
    if attrs["end"] != -1:
        raise ValueError(f"Shape with end={attrs['end']} is not supported")
    if attrs["start"] != 0:
        raise ValueError(f"Shape with start={attrs['start']} is not supported")
    return attrs


def get_attrs_transpose(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract Transpose operator attributes."""
    attrs = scan_attrs({"perm": None}, node.attribute)
    if attrs["perm"] is None:
        raise ValueError("Transpose perm is required")
    return attrs


EXTRACT_ATTRS_MAP: dict[str, Callable[[NodeProto, dict[str, TensorProto]], dict[str, Any]]] = {
    "ArgMax": get_attrs_argmax,
    "AveragePool": get_attrs_avgpool,
    "BatchNormalization": get_attrs_batchnorm,
    "Cast": get_attrs_cast,
    "Concat": get_attrs_concat,
    "Conv": get_attrs_conv,
    "Constant": get_attrs_constant,
    "ConvTranspose": get_attrs_conv_transpose,
    "ConstantOfShape": get_attrs_constantofshape,
    "Elu": get_attrs_simple({"alpha": 1.0}),
    "Flatten": get_attrs_simple({"axis": 1}),
    "Gather": get_attrs_simple({"axis": 0}),
    "Gelu": get_attrs_simple({"approximate": "none"}),
    "Gemm": get_attrs_simple({"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}),
    "LeakyRelu": get_attrs_simple({"alpha": 0.01}),
    "MaxPool": get_attrs_maxpool,
    "Pad": get_attrs_simple({"mode": "constant"}),
    "ReduceMean": get_attrs_reduce("ReduceMean"),
    "ReduceSum": get_attrs_reduce("ReduceSum"),
    "Reshape": get_attrs_reshape,
    "Resize": get_attrs_resize,
    "Shape": get_attrs_shape,
    "Scatter": get_attrs_scatter,
    "ScatterElements": get_attrs_scatterelement,
    "ScatterND": get_attrs_scatternd,
    "Softmax": get_attrs_simple({"axis": -1}),
    "Split": get_attrs_simple({"axis": 0, "num_outputs": None}),
    "Transpose": get_attrs_transpose,
    "Unsqueeze": get_attrs_simple({"axes": None}),
    "Upsample": get_attrs_simple({"mode": "nearest"}),
}


def get_onnx_attrs(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """
    Extract attributes from an ONNX node.

    :param node: ONNX node
    :param initializers: Model initializers
    :return: Dictionary of extracted attributes
    """
    extractor = EXTRACT_ATTRS_MAP.get(node.op_type)
    if extractor is None:
        raise NotImplementedError(f"Unsupported operator: {node.op_type}")
    return extractor(node, initializers)
