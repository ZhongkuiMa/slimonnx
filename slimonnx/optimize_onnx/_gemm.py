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
                    array = array.T
                    transA = 0
                if alpha != 1.0:
                    array = array * alpha
                    alpha = 1.0
                new_initializer = onnx.numpy_helper.from_array(array, node.input[0])
                initializers[node.input[0]] = new_initializer

            if node.input[1] in initializers:
                initializer = initializers[node.input[1]]
                array = onnx.numpy_helper.to_array(initializer)
                if transB == 1:
                    array = array.T
                    transB = 0
                if alpha != 1.0:
                    array = array * alpha
                    alpha = 1.0
                new_initializer = onnx.numpy_helper.from_array(array, node.input[1])
                initializers[node.input[1]] = new_initializer

            if node.input[2] in initializers:
                initializer = initializers[node.input[2]]
                array = onnx.numpy_helper.to_array(initializer)
                if beta != 1.0:
                    array = array * beta
                    beta = 1.0
                new_initializer = onnx.numpy_helper.from_array(array, node.input[2])
                initializers[node.input[2]] = new_initializer

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

            count += 1

    if utils.VERBOSE:
        print(f"Simplify {count} Gemm nodes.")

    return nodes
