__docformat__ = "restructuredtext"
__all__ = ["_simplify_gemm"]

import onnx.numpy_helper
from onnx import NodeProto, TensorProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from ..onnx_attrs import get_onnx_attrs


def _simplify_gemm(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    count = 0
    new_nodes = []
    # Here we encounter a problem:
    # Several Gemm nodes use the same initializer and if we change the initializer
    # it will change the value of the other Gemm nodes.
    # So we need to create a new copy of the initializer for each Gemm node.
    for node in nodes:
        if node.op_type == "Gemm":
            attrs = get_onnx_attrs(node, initializers)
            alpha = attrs["alpha"]
            beta = attrs["beta"]
            transA = attrs["transA"]
            transB = attrs["transB"]
            if node.input[0] in initializers:
                initializer = initializers[node.input[0]]
                array = onnx.numpy_helper.to_array(initializer)
                if transA == 1:
                    array = array.copy().T
                    transA = 0
                if alpha != 1.0:
                    array = array * alpha
                    alpha = 1.0
                new_initer_name = node.input[0] + "_" + str(count)
                new_initer = onnx.numpy_helper.from_array(array, new_initer_name)
                initializers[new_initer_name] = new_initer
                node.input[0] = new_initer_name

            if node.input[1] in initializers:
                initializer = initializers[node.input[1]]
                array = onnx.numpy_helper.to_array(initializer)
                if transB == 1:
                    array = array.copy().T
                    transB = 0
                if alpha != 1.0:
                    array = array * alpha
                    alpha = 1.0
                new_initer_name = node.input[1] + "_" + str(count)
                new_initer = onnx.numpy_helper.from_array(array, new_initer_name)
                initializers[new_initer_name] = new_initer
                node.input[1] = new_initer_name

            if node.input[2] in initializers:
                initializer = initializers[node.input[2]]
                array = onnx.numpy_helper.to_array(initializer)
                if beta != 1.0:
                    array = array * beta
                    beta = 1.0
                new_initer_name = node.input[2] + "_" + str(count)
                new_initer = onnx.numpy_helper.from_array(array, new_initer_name)
                initializers[new_initer_name] = new_initer
                node.input[2] = new_initer_name

            # Change the attributes of the node
            for i, attr in enumerate(node.attribute):
                if attr.name == "alpha":
                    node.attribute[i].f = alpha
                elif attr.name == "beta":
                    node.attribute[i].f = beta
                elif attr.name == "transA":
                    node.attribute[i].i = transA
                elif attr.name == "transB":
                    node.attribute[i].i = transB

            # Create a new gemm node to make the following conditions:
            # 1. If there is only one variable in the input, make it be the first input
            # 2. Remove all defalt values of the attributes
            var_name = node.input[0]
            weight_name = node.input[1]
            if var_name in initializers and weight_name not in initializers:
                var_name, weight_name = weight_name, var_name
            input_names = [var_name, weight_name]
            if len(node.input) == 3:
                input_names.append(node.input[2])
            node = NodeProto(
                name=node.name,
                op_type="Gemm",
                input=input_names,
                output=node.output,
            )

            count += 1
        new_nodes.append(node)
    nodes = new_nodes

    # Remove the unused initializers because we create copy of the initializers in the
    # above loop
    all_input_names = [input_name for node in nodes for input_name in node.input]
    for name in list(initializers.keys()):
        if name not in all_input_names:
            del initializers[name]

    if utils.VERBOSE:
        print(f"Simplify {count} Gemm nodes.")

    return nodes
