__docformat__ = ["restructuredtext"]
__all__ = ["_fuse_transpose_batchnorm_transpose"]

import numpy as np
import onnx
from onnx import NodeProto, TensorProto

from ._utils import *
from ..onnx_attrs import get_onnx_attrs


def _fuse_transpose_batchnorm_transpose(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
) -> list[NodeProto]:
    count = 0

    new_nodes = []
    pre_pre_node = None
    pre_node = None
    for node in nodes:
        # print(node.op_type, node.input, node.output)
        new_node = node
        if (
            node.op_type == "Transpose"
            and pre_pre_node is not None
            and pre_node is not None
            and pre_pre_node.op_type == "Transpose"
            and pre_node.op_type == "BatchNormalization"
            and _in_single_path(pre_pre_node, pre_node, nodes)
            and _in_single_path(pre_node, node, nodes)
        ):
            new_nodes.pop()
            new_nodes.pop()
            tp_node1, bn_node, tp_node2 = pre_pre_node, pre_node, node
            data_type = initializers[bn_node.input[2]].data_type

            perm1 = get_onnx_attrs(tp_node1, initializers)["perm"]
            perm2 = get_onnx_attrs(tp_node2, initializers)["perm"]
            _mode = (0, 2, 1)
            assert all(p_i == p_j for p_i, p_j in zip(perm1, _mode))
            assert all(p_i == p_j for p_i, p_j in zip(perm2, _mode))

            epsilon, scale, b, mean, var = _get_batchnorm_params(bn_node, initializers)

            weight_name = bn_node.input[1] + "_gemm"
            bias_name = bn_node.input[2] + "_gemm"
            bn_weight = scale / np.sqrt(var + epsilon)
            weight = np.diag(bn_weight)
            bias = b - mean * bn_weight
            weight = onnx.helper.make_tensor(
                name=weight_name,
                data_type=data_type,
                dims=weight.shape,
                vals=weight.flatten().tolist(),
            )
            bias = onnx.helper.make_tensor(
                name=bias_name,
                data_type=data_type,
                dims=bias.shape,
                vals=bias.flatten().tolist(),
            )
            initializers[weight_name] = weight
            initializers[bias_name] = bias

            new_node = onnx.helper.make_node(
                op_type="Gemm",
                inputs=[tp_node1.input[0], weight_name, bias_name],
                outputs=tp_node2.output,
                name=bn_node.name + "_gemm",
            )

            count += 1

        new_nodes.append(new_node)
        pre_pre_node = pre_node
        pre_node = node

    print(f"Fused {count} Transpose-BN-Transpose nodes.")

    return new_nodes
