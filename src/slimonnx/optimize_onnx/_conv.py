__docformat__ = "restructuredtext"
__all__ = ["_simplify_conv_to_flatten_gemm"]

import onnx
from onnx import NodeProto, TensorProto

from slimonnx.optimize_onnx._utils import _get_conv_params, _is_only_next_node


def _can_simplify_conv(
    weight: TensorProto,
    pre_output_shape: list[int],
    pads: list[int],
    strides: list[int],
    dilations: list[int],
    group: int,
    auto_pad: str,
) -> bool:
    """Check if a Conv node can be simplified to Flatten+Gemm.

    :param weight: Conv weight tensor
    :param pre_output_shape: Shape of previous conv output
    :param pads: Padding values
    :param strides: Stride values
    :param dilations: Dilation values
    :param group: Group value
    :param auto_pad: Auto padding mode
    :return: True if conv can be simplified
    """
    input_channel = weight.shape[1]
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]

    return (
        pre_output_shape[1] == input_channel
        and pre_output_shape[2] == kernel_height
        and pre_output_shape[3] == kernel_width
        and all(p == 0 for p in pads)
        and all(s == 1 for s in strides)
        and all(d == 1 for d in dilations)
        and group == 1
        and auto_pad == "NOTSET"
    )


def _validate_conv_simplification(
    pre_pre_node: NodeProto | None, pre_node: NodeProto, node: NodeProto, nodes: list[NodeProto]
) -> None:
    """Validate that Conv simplification is safe.

    :param pre_pre_node: Node before pre_node
    :param pre_node: Previous node (Conv to be replaced)
    :param node: Current node
    :param nodes: All nodes
    :raises ValueError: If simplification is invalid
    """
    if pre_pre_node is None:
        raise ValueError(
            f"Conv node {pre_node.name} marked for simplification but has no predecessor."
        )
    if not _is_only_next_node(pre_pre_node, pre_node, nodes):
        raise ValueError(
            f"Conv simplification invalid: {pre_pre_node.name} has multiple successors."
        )
    if not _is_only_next_node(pre_node, node, nodes):
        raise ValueError(f"Conv simplification invalid: {pre_node.name} has multiple successors.")


def _create_flatten_gemm_nodes(
    conv_node: NodeProto, weight: TensorProto, initializers: dict[str, TensorProto]
) -> tuple[NodeProto, NodeProto]:
    """Create Flatten and Gemm nodes to replace Conv.

    :param conv_node: Original Conv node
    :param weight: Conv weight tensor
    :param initializers: Initializers dictionary
    :return: Tuple of (flatten_node, gemm_node)
    """
    # Flatten the weight and store the new weight in the initializers
    weight_reshaped = weight.reshape(weight.shape[0], -1).T
    new_weight_initer = onnx.numpy_helper.from_array(weight_reshaped, conv_node.input[1])
    initializers[conv_node.input[1]] = new_weight_initer

    # Create a flatten node
    middle_output_name = conv_node.output[0] + "_flatten"
    flatten_node = NodeProto(
        name=conv_node.name + "_flatten",
        op_type="Flatten",
        input=[conv_node.input[0]],
        output=[middle_output_name],
    )

    # Create a new Gemm node
    gemm_node = NodeProto(
        name=conv_node.name + "_flatten_gemm",
        op_type="Gemm",
        input=[middle_output_name, conv_node.input[1], conv_node.input[2]],
        output=[conv_node.output[0]],
    )

    return flatten_node, gemm_node


def _simplify_conv_to_flatten_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, int | list[int]],
) -> list[NodeProto]:
    pre_conv_node = None
    conv_nodes_to_replaced_names = []
    conv_nodes_to_replaced_args = {}
    for node in nodes:
        if node.op_type == "Conv":
            conv_node = node
            if pre_conv_node is not None:
                (weight, _bias, attrs) = _get_conv_params(
                    node, initializers, remove_initializers=False
                )
                if weight.ndim != 4:
                    raise ValueError(
                        f"Conv node {node.name} has unsupported weight dimension {weight.ndim}D. "
                        "Only 4D weight is supported."
                    )

                pre_output_shape_raw = data_shapes[pre_conv_node.output[0]]
                # Convert int to list[int] if needed
                pre_output_shape = (
                    [pre_output_shape_raw]
                    if isinstance(pre_output_shape_raw, int)
                    else pre_output_shape_raw
                )

                if _can_simplify_conv(
                    weight,
                    pre_output_shape,
                    attrs["pads"],
                    attrs["strides"],
                    attrs["dilations"],
                    attrs["group"],
                    attrs.get("auto_pad", "NOTSET"),
                ):
                    conv_nodes_to_replaced_names.append(conv_node.name)
                    conv_nodes_to_replaced_args[node.name] = weight

            pre_conv_node = conv_node

    # We need the second loop to avoid change the previous node to a non-Conv node.
    new_nodes = []
    pre_pre_node = None
    pre_node = None

    for node in nodes:
        if pre_node is not None and pre_node.name in conv_nodes_to_replaced_names:
            _validate_conv_simplification(pre_pre_node, pre_node, node, nodes)

            weight = conv_nodes_to_replaced_args[pre_node.name]

            # Pop the Conv node that is the pre_node
            new_nodes.pop()

            flatten_node, gemm_node = _create_flatten_gemm_nodes(pre_node, weight, initializers)
            new_nodes.append(flatten_node)
            new_nodes.append(gemm_node)

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    return new_nodes
