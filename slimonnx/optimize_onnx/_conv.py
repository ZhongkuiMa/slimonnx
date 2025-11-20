__docformat__ = "restructuredtext"
__all__ = ["_simplify_conv_to_flatten_gemm"]

import onnx
from onnx import NodeProto, TensorProto


from ._utils import _is_only_next_node, _get_conv_params


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
                (weight, bias, attrs) = _get_conv_params(node, initializers, False)
                kernel_shape = attrs["kernel_shape"]
                pads = attrs["pads"]
                strides = attrs["strides"]
                dilations = attrs["dilations"]
                group = attrs["group"]
                auto_pad = attrs.get("auto_pad", "NOTSET")
                if weight.ndim != 4:
                    raise ValueError(
                        f"Conv node {node.name} has unsupported weight dimension {weight.ndim}D. Only 4D weight is supported."
                    )
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
            if pre_pre_node is None:
                raise ValueError(
                    f"Conv node {pre_node.name} marked for simplification but has no predecessor."
                )
            if not _is_only_next_node(pre_pre_node, pre_node, nodes):
                raise ValueError(
                    f"Conv simplification invalid: {pre_pre_node.name} has multiple successors."
                )
            if not _is_only_next_node(pre_node, node, nodes):
                raise ValueError(
                    f"Conv simplification invalid: {pre_node.name} has multiple successors."
                )
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

    return new_nodes
