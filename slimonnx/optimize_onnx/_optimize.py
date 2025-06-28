__docformat__ = ["restructuredtext"]
__all__ = ["optimize_onnx"]

import onnx
from onnx import ModelProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from slimonnx.shapeonnx.shapeonnx.infer_shape import infer_onnx_shape
from ._bn_conv import *
from ._bn_gemm import *
from ._bn_transpose import *
from ._conv import *
from ._cst2initer import *
from ._cst_op import *
from ._gemm import *
from ._gemm_gemm import *
from ._mm_add import *
from ._name import *
from ._ordering import *
from ._redundant import *
from ..utils import *


def optimize_onnx(
    model: ModelProto,
    constant_to_initializer: bool = False,
    fuse_constant_nodes: bool = False,
    fuse_matmul_add: bool = False,
    fuse_gemm_reshape_bn: bool = False,
    fuse_bn_reshape_gemm: bool = False,
    fuse_bn_gemm: bool = False,
    fuse_transpose_bn_transpose: bool = False,
    fuse_gemm_gemm: bool = False,
    fuse_conv_bn: bool = False,
    fuse_bn_conv: bool = False,
    fuse_convtransposed_bn: bool = False,
    fuse_bn_convtransposed: bool = False,
    simplify_conv_to_flatten_gemm: bool = False,
    simplify_gemm: bool = False,
    remove_redundant_operations: bool = False,
    reorder_by_strict_topological_order: bool = False,
    simplify_node_name: bool = False,
    verbose: bool = False,
) -> ModelProto:

    utils.VERBOSE = verbose

    if verbose:
        print("Clear ONNX docstring.")
    clear_onnx_docstring(model)

    graph_name = model.graph.name + "_slimmed"

    if verbose:
        print("Set batch size to 1.")
    input_nodes = get_input_nodes(model)
    output_nodes = get_output_nodes(model)
    initializers = get_initializers(model)

    nodes = list(model.graph.node)

    # DEBUG
    # if simplify_node_name:
    #     # The name is ordered.
    #     nodes, initializers = _simplify_names(
    #         input_nodes, output_nodes, nodes, initializers
    #     )

    # NOTE: Some operations need calculating the shape of nodes.
    # To avoid some operations has changed the shape of nodes, we need to calculate
    # again the shape of nodes before the operations.
    if constant_to_initializer:
        nodes = _constant_to_initializer(nodes, initializers)
    if fuse_constant_nodes:
        data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers)
        nodes, initializers = _fuse_constant_nodes(nodes, initializers, data_shapes)
    if remove_redundant_operations:
        data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers)
        # We need the correct order to check the redundant operations.
        nodes = _reorder_by_strict_topological_order(nodes)
        nodes = _remove_redundant_operations(
            nodes, initializers, data_shapes, output_nodes
        )
    if simplify_conv_to_flatten_gemm:
        data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers)
        nodes = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)
    if fuse_matmul_add:
        nodes = _fuse_matmul_add(nodes, initializers)
    if fuse_gemm_reshape_bn:
        nodes = _fuse_gemm_reshape_bn(nodes, initializers)
    if fuse_bn_reshape_gemm:
        nodes = _fuse_bn_reshape_gemm(nodes, initializers)
    if fuse_bn_gemm:
        nodes = _fuse_bn_gemm(nodes, initializers)
    if fuse_transpose_bn_transpose:
        nodes = _fuse_transpose_batchnorm_transpose(nodes, initializers)
    if simplify_gemm:
        nodes = _simplify_gemm(nodes, initializers)
    if fuse_gemm_gemm:
        # There may be multiple gemm nodes, so we need to fuse them multiple times.
        nodes = _fuse_gemm_gemm(nodes, initializers)
        nodes = _fuse_gemm_gemm(nodes, initializers)

    if fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers)
    if fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
    if fuse_convtransposed_bn:
        nodes = _fuse_convtranspose_bn_or_bn_convtranspose(nodes, initializers)
    if fuse_bn_convtransposed:
        nodes = _fuse_convtranspose_bn_or_bn_convtranspose(
            nodes, initializers, is_convtranspose_bn=False
        )
    if reorder_by_strict_topological_order:
        # There maybe repeated named nodes, so we need to simplify the names first
        nodes, initializers = _simplify_names(
            input_nodes, output_nodes, nodes, initializers
        )
        nodes = _reorder_by_strict_topological_order(nodes)
    if simplify_node_name:
        # The name is ordered.
        nodes, initializers = _simplify_names(
            input_nodes, output_nodes, nodes, initializers
        )

    if verbose:
        print("Assembly new model...")

    new_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            graph_name,
            input_nodes,
            output_nodes,
            list(initializers.values()),
        )
    )

    return new_model
