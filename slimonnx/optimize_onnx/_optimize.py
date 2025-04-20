__docformat__ = ["restructuredtext"]
__all__ = ["optimize_onnx"]

import onnx
from onnx import ModelProto

import slimonnx.slimonnx.optimize_onnx._utils as utils
from slimonnx.shapeonnx.shapeonnx.infer_shape import infer_onnx_shape
from ._bn_conv import *
from ._bn_gemm import *
from ._bn_transpose import *
from ._cst2initer import *
from ._gemm_gemm import *
from ._mm_add import *
from ._ordering import *
from ._rm_redundant import *
from ._shp2initer import *
from ._sim_name import *
from ..utils import *


def optimize_onnx(
    model: ModelProto,
    constant_to_initializer: bool = False,
    shape_to_initializer: bool = False,
    fuse_matmul_add: bool = False,
    fuse_gemm_reshape_bn: bool = False,
    fuse_bn_reshape_gemm: bool = False,
    fuse_bn_gemm: bool = False,
    fuse_transpose_bn_transpose: bool = False,
    fuse_gemm_gemm: bool = False,
    fuse_conv_bn: bool = False,
    fuse_bn_conv: bool = False,
    fuse_convtransposed_bn: bool = False,
    remove_redundant_reshape: bool = False,
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

    data_shapes = None

    if constant_to_initializer:
        nodes = _constant_to_initializer(nodes, initializers)
    if shape_to_initializer:
        data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers)
        nodes = _shape_to_initializer(nodes, initializers, data_shapes)
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
    if fuse_gemm_gemm:
        nodes = _fuse_gemm_gemm(nodes, initializers)
    if fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=True)
    if fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)
    if fuse_convtransposed_bn:
        nodes = _fuse_convtranspose_bn(nodes, initializers)
    if remove_redundant_reshape:
        if data_shapes is None:
            data_shapes = infer_onnx_shape(
                input_nodes, output_nodes, nodes, initializers
            )
        nodes = _remove_redundant_reshape(nodes, initializers, data_shapes)
    if remove_redundant_operations:
        nodes = _remove_redundant_operations(nodes, initializers)
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
