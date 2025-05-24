__docformat__ = "restructuredtext"
__all__ = ["_simplify_conv_to_flatten_gemm"]

import onnx
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ._utils import *


def _simplify_conv_to_flatten_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    data_shapes: dict[str, list[int]],
) -> list[NodeProto]:
    count = 0

    pre_conv_node = None
    conv_nodes_to_replaced_names = []
    conv_nodes_to_replaced_args = {}
    for node in nodes:
        if node.op_type == "Conv":
            conv_node = node
            if pre_conv_node is not None:
                (
                    kernel_shape,
                    pads,
                    strides,
                    dilations,
                    group,
                    auto_pad,
                    weight,
                    bias,
                ) = _get_conv_params(node, initializers, False)
                assert weight.ndim == 4, f"{weight.ndim}D weight is not supported."
                input_channel = weight.shape[1]
                kernel_height = weight.shape[2]
                kernel_width = weight.shape[3]

                pre_output_shape = data_shapes[pre_conv_node.output[0]]

                if (
                    pre_output_shape[1] == input_channel
                    and pre_output_shape[2] == kernel_height
                    and pre_output_shape[3] == kernel_width
                    and all(p == 0 for p in pads)
                    and all(s == 1 for s in strides)
                    and all(d == 1 for d in dilations)
                    and group == 1
                    and auto_pad == "NOTSET"
                ):
                    conv_nodes_to_replaced_names.append(conv_node.name)
                    conv_nodes_to_replaced_args[node.name] = (
                        kernel_shape,
                        pads,
                        strides,
                        dilations,
                        group,
                        auto_pad,
                        weight,
                        bias,
                    )

            pre_conv_node = conv_node

    # We need the second loop to avoid change the previous node to a non-Conv node.
    new_nodes = []
    pre_pre_node = None
    pre_node = None

    for node in nodes:
        if pre_node is not None and pre_node.name in conv_nodes_to_replaced_names:
            assert pre_pre_node is not None
            assert _is_only_next_node(pre_pre_node, pre_node, nodes)
            assert _is_only_next_node(pre_node, node, nodes)
            count += 1

            kernel_shape, pads, strides, dilations, group, auto_pad, weight, bias = (
                conv_nodes_to_replaced_args[pre_node.name]
            )

            # Flatten the weight and store the new weight in the initializers
            # The bias has no need to be flattened
            weight = weight.reshape(weight.shape[0], -1)
            weight = weight.T
            new_weight_initer = onnx.numpy_helper.from_array(weight, pre_node.input[1])
            initializers[pre_node.input[1]] = new_weight_initer

            # Pop the Conv node that is the pre_node
            new_nodes.pop()

            # Create a flatten node
            midlle_output_name = pre_node.output[0] + "_flatten"
            flatten_node = NodeProto(
                name=pre_node.name + "_flatten",
                op_type="Flatten",
                input=[pre_node.input[0]],
                output=[midlle_output_name],
            )
            new_nodes.append(flatten_node)

            # Create a new Gemm node
            gemm_node = NodeProto(
                name=pre_node.name + "_flatten_gemm",
                op_type="Gemm",
                input=[midlle_output_name, pre_node.input[1], pre_node.input[2]],
                output=[pre_node.output[0]],
            )

            new_nodes.append(gemm_node)

        pre_pre_node = pre_node
        pre_node = node
        new_nodes.append(node)

    if utils.VERBOSE:
        print(f"Simplify {count} Conv to Flatten-Gemm nodes.")

    return new_nodes
