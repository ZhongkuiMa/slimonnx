"""Main optimization orchestration for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = ["optimize_onnx"]

import onnx
from onnx import ModelProto

from shapeonnx.shapeonnx.infer_shape import infer_onnx_shape
from ._bn_conv import (
    _fuse_conv_bn_or_bn_conv,
    _fuse_convtranspose_bn_or_bn_convtranspose,
)
from ._depthwise_conv import _fuse_depthwise_conv_bn_or_bn_depthwise_conv
from ._bn_gemm import _fuse_gemm_reshape_bn, _fuse_bn_reshape_gemm, _fuse_bn_gemm
from ._bn_transpose import _fuse_transpose_batchnorm_transpose
from ._conv import _simplify_conv_to_flatten_gemm
from ._cst2initer import _constant_to_initializer
from ._cst_op import _fuse_constant_nodes
from ._dropout import remove_dropout as _remove_dropout
from ._gemm import _simplify_gemm
from ._gemm_gemm import _fuse_gemm_gemm
from ._mm_add import _fuse_matmul_add
from ._name import _simplify_names
from ._ordering import _reorder_by_strict_topological_order
from ._redundant import _remove_redundant_operations
from ..utils import (
    clear_onnx_docstring,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)


def optimize_onnx(
    model: ModelProto,
    constant_to_initializer: bool = False,
    constant_folding: bool = False,
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
    fuse_depthwise_conv_bn: bool = False,
    fuse_bn_depthwise_conv: bool = False,
    simplify_conv_to_flatten_gemm: bool = False,
    simplify_gemm: bool = False,
    remove_dropout: bool = True,
    remove_redundant_operations: bool = False,
    reorder_by_strict_topological_order: bool = False,
    simplify_node_name: bool = False,
    has_batch_dim: bool = True,
    verbose: bool = False,
) -> ModelProto:
    """Optimize ONNX model with various fusion and simplification passes.

    :param model: Input ONNX model
    :param constant_to_initializer: Convert constant nodes to initializers
    :param constant_folding: Fold constant operations (renamed from fuse_constant_nodes)
    :param fuse_matmul_add: Fuse MatMul + Add to Gemm
    :param fuse_gemm_reshape_bn: Fuse Gemm-Reshape-BatchNorm
    :param fuse_bn_reshape_gemm: Fuse BatchNorm-Reshape-Gemm
    :param fuse_bn_gemm: Fuse BatchNorm-Gemm
    :param fuse_transpose_bn_transpose: Fuse Transpose-BN-Transpose
    :param fuse_gemm_gemm: Fuse consecutive Gemm nodes
    :param fuse_conv_bn: Fuse Conv-BatchNorm
    :param fuse_bn_conv: Fuse BatchNorm-Conv
    :param fuse_convtransposed_bn: Fuse ConvTranspose-BatchNorm
    :param fuse_bn_convtransposed: Fuse BatchNorm-ConvTranspose
    :param fuse_depthwise_conv_bn: Fuse Depthwise Conv-BatchNorm
    :param fuse_bn_depthwise_conv: Fuse BatchNorm-Depthwise Conv
    :param simplify_conv_to_flatten_gemm: Convert Conv to Flatten-Gemm
    :param simplify_gemm: Normalize Gemm attributes
    :param remove_dropout: Remove Dropout nodes (default True for inference)
    :param remove_redundant_operations: Remove no-op nodes
    :param reorder_by_strict_topological_order: Topological sort
    :param simplify_node_name: Rename nodes sequentially
    :param has_batch_dim: Whether model has batch dimension
    :param verbose: Print progress messages
    :return: Optimized ONNX model
    """
    graph_name = model.graph.name + "_slimmed"

    model = clear_onnx_docstring(model)

    # TODO: Here, we can use extract_nodes()
    initializers = get_initializers(model, verbose)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim, verbose)
    output_nodes = get_output_nodes(model, has_batch_dim, verbose)

    nodes = list(model.graph.node)

    if constant_to_initializer:
        nodes = _constant_to_initializer(nodes, initializers, verbose)

    # Update model with converted constants before any shape inference
    if constant_to_initializer:
        model.graph.ClearField("node")
        model.graph.node.extend(nodes)
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(list(initializers.values()))

    if remove_dropout:
        if verbose:
            print("Removing Dropout nodes...")
        model = _remove_dropout(model)
        nodes = list(model.graph.node)

    if constant_folding:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes, initializers = _fuse_constant_nodes(
            nodes, initializers, data_shapes, verbose
        )
    if remove_redundant_operations:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _reorder_by_strict_topological_order(nodes)
        nodes = _remove_redundant_operations(
            nodes, initializers, data_shapes, output_nodes, verbose
        )
    if simplify_conv_to_flatten_gemm:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _simplify_conv_to_flatten_gemm(
            nodes, initializers, data_shapes, verbose
        )
    if fuse_matmul_add:
        nodes = _fuse_matmul_add(nodes, initializers, verbose)
    if fuse_gemm_reshape_bn:
        nodes = _fuse_gemm_reshape_bn(nodes, initializers, verbose)
    if fuse_bn_reshape_gemm:
        nodes = _fuse_bn_reshape_gemm(nodes, initializers, verbose)
    if fuse_bn_gemm:
        nodes = _fuse_bn_gemm(nodes, initializers, verbose)
    if fuse_transpose_bn_transpose:
        nodes = _fuse_transpose_batchnorm_transpose(nodes, initializers, verbose)
    if simplify_gemm:
        nodes = _simplify_gemm(nodes, initializers, verbose)
    if fuse_gemm_gemm:
        nodes = _fuse_gemm_gemm(nodes, initializers, verbose)
        nodes = _fuse_gemm_gemm(nodes, initializers, verbose)

    if fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, verbose=verbose)
    if fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(
            nodes, initializers, is_conv_bn=False, verbose=verbose
        )
    if fuse_convtransposed_bn:
        nodes = _fuse_convtranspose_bn_or_bn_convtranspose(
            nodes, initializers, verbose=verbose
        )
    if fuse_bn_convtransposed:
        nodes = _fuse_convtranspose_bn_or_bn_convtranspose(
            nodes, initializers, is_convtranspose_bn=False, verbose=verbose
        )
    if fuse_depthwise_conv_bn:
        nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, verbose=verbose
        )
    if fuse_bn_depthwise_conv:
        nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(
            nodes, initializers, is_conv_bn=False, verbose=verbose
        )
    if reorder_by_strict_topological_order:
        nodes, initializers = _simplify_names(
            input_nodes, output_nodes, nodes, initializers
        )
        nodes = _reorder_by_strict_topological_order(nodes)
    if simplify_node_name:
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
