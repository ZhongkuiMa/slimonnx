"""Main optimization orchestration for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = ["optimize_onnx"]

import onnx
from onnx import ModelProto, NodeProto, TensorProto
from shapeonnx.infer_shape import infer_onnx_shape

from slimonnx.optimize_onnx._bn_conv import (
    _fuse_conv_bn_or_bn_conv,
    _fuse_conv_transpose_bn_or_bn_conv_transpose,
)
from slimonnx.optimize_onnx._bn_gemm import (
    _fuse_bn_gemm,
    _fuse_bn_reshape_gemm,
    _fuse_gemm_reshape_bn,
)
from slimonnx.optimize_onnx._bn_transpose import _fuse_transpose_batchnorm_transpose
from slimonnx.optimize_onnx._conv import _simplify_conv_to_flatten_gemm
from slimonnx.optimize_onnx._cst2initer import _constant_to_initializer
from slimonnx.optimize_onnx._cst_op import _fuse_constant_nodes
from slimonnx.optimize_onnx._depthwise_conv import (
    _fuse_depthwise_conv_bn_or_bn_depthwise_conv,
)
from slimonnx.optimize_onnx._dropout import remove_dropout as _remove_dropout
from slimonnx.optimize_onnx._gemm import _simplify_gemm
from slimonnx.optimize_onnx._gemm_gemm import _fuse_gemm_gemm
from slimonnx.optimize_onnx._mm_add import _fuse_matmul_add
from slimonnx.optimize_onnx._name import _simplify_names
from slimonnx.optimize_onnx._ordering import _reorder_by_strict_topological_order
from slimonnx.optimize_onnx._redundant import _remove_redundant_operations
from slimonnx.optimize_onnx._reshape import _resolve_reshape_negative_one
from slimonnx.utils import (
    clear_onnx_docstring,
    get_initializers,
    get_input_nodes,
    get_output_nodes,
)


def _apply_shape_based_optimizations(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list,
    output_nodes: list,
    has_batch_dim: bool,
    constant_folding: bool,
    remove_redundant_operations: bool,
    simplify_conv_to_flatten_gemm: bool,
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Apply optimizations that require shape inference."""
    if constant_folding:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes, initializers = _fuse_constant_nodes(nodes, initializers, data_shapes)

    data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers, has_batch_dim)
    nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)

    if remove_redundant_operations:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _reorder_by_strict_topological_order(nodes)
        nodes = _remove_redundant_operations(nodes, initializers, data_shapes, output_nodes)

    if simplify_conv_to_flatten_gemm:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

    return nodes, initializers


def _apply_gemm_bn_fusions(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list,
    output_nodes: list,
    has_batch_dim: bool,
    fuse_matmul_add: bool,
    fuse_gemm_reshape_bn: bool,
    fuse_bn_reshape_gemm: bool,
    fuse_bn_gemm: bool,
    fuse_transpose_bn_transpose: bool,
    simplify_gemm: bool,
    fuse_gemm_gemm: bool,
) -> list[NodeProto]:
    """Apply Gemm and BatchNorm fusion optimizations."""
    if fuse_matmul_add:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _fuse_matmul_add(nodes, initializers, input_nodes, data_shapes)

    if fuse_gemm_reshape_bn:
        nodes = _fuse_gemm_reshape_bn(nodes, initializers)
    if fuse_bn_reshape_gemm:
        nodes = _fuse_bn_reshape_gemm(nodes, initializers)
    if fuse_bn_gemm:
        nodes = _fuse_bn_gemm(nodes, initializers)

    if fuse_transpose_bn_transpose:
        data_shapes = infer_onnx_shape(
            input_nodes, output_nodes, nodes, initializers, has_batch_dim
        )
        nodes = _fuse_transpose_batchnorm_transpose(nodes, initializers, input_nodes, data_shapes)

    if simplify_gemm:
        nodes = _simplify_gemm(nodes, initializers)

    if fuse_gemm_gemm:
        nodes = _fuse_gemm_gemm(nodes, initializers)
        nodes = _fuse_gemm_gemm(nodes, initializers)

    return nodes


def _apply_conv_fusions(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    fuse_conv_bn: bool,
    fuse_bn_conv: bool,
    fuse_conv_transpose_bn: bool,
    fuse_bn_conv_transpose: bool,
    fuse_depthwise_conv_bn: bool,
    fuse_bn_depthwise_conv: bool,
) -> list[NodeProto]:
    """Apply Conv and BatchNorm fusion optimizations."""
    if fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers)
    if fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)

    if fuse_conv_transpose_bn:
        nodes = _fuse_conv_transpose_bn_or_bn_conv_transpose(nodes, initializers)
    if fuse_bn_conv_transpose:
        nodes = _fuse_conv_transpose_bn_or_bn_conv_transpose(
            nodes, initializers, is_conv_transpose_bn=False
        )

    if fuse_depthwise_conv_bn:
        nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers)
    if fuse_bn_depthwise_conv:
        nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers, is_conv_bn=False)

    return nodes


def optimize_onnx(
    model: ModelProto,
    constant_folding: bool = False,
    fuse_matmul_add: bool = False,
    fuse_gemm_reshape_bn: bool = False,
    fuse_bn_reshape_gemm: bool = False,
    fuse_bn_gemm: bool = False,
    fuse_transpose_bn_transpose: bool = False,
    fuse_gemm_gemm: bool = False,
    fuse_conv_bn: bool = False,
    fuse_bn_conv: bool = False,
    fuse_conv_transpose_bn: bool = False,
    fuse_bn_conv_transpose: bool = False,
    fuse_depthwise_conv_bn: bool = False,
    fuse_bn_depthwise_conv: bool = False,
    simplify_conv_to_flatten_gemm: bool = False,
    simplify_gemm: bool = False,
    remove_dropout: bool = True,
    remove_redundant_operations: bool = False,
    reorder_by_strict_topological_order: bool = False,
    simplify_node_name: bool = False,
    has_batch_dim: bool = True,
) -> ModelProto:
    """Optimize ONNX model with various fusion and simplification passes.

    Constants are always converted to initializers for shape inference.

    :param model: Input ONNX model
    :param constant_folding: Fold constant operations (renamed from fuse_constant_nodes)
    :param fuse_matmul_add: Fuse MatMul + Add to Gemm
    :param fuse_gemm_reshape_bn: Fuse Gemm-Reshape-BatchNorm
    :param fuse_bn_reshape_gemm: Fuse BatchNorm-Reshape-Gemm
    :param fuse_bn_gemm: Fuse BatchNorm-Gemm
    :param fuse_transpose_bn_transpose: Fuse Transpose-BN-Transpose
    :param fuse_gemm_gemm: Fuse consecutive Gemm nodes
    :param fuse_conv_bn: Fuse Conv-BatchNorm
    :param fuse_bn_conv: Fuse BatchNorm-Conv (skips cases with padding)
    :param fuse_bn_conv_with_padding: Fuse BatchNorm-Conv with padding by inserting Pad node
    :param fuse_conv_transposed_bn: Fuse ConvTranspose-BatchNorm
    :param fuse_bn_conv_transposed: Fuse BatchNorm-ConvTranspose
    :param fuse_depthwise_conv_bn: Fuse Depthwise Conv-BatchNorm
    :param fuse_bn_depthwise_conv: Fuse BatchNorm-Depthwise Conv
    :param simplify_conv_to_flatten_gemm: Convert Conv to Flatten-Gemm
    :param simplify_gemm: Normalize Gemm attributes
    :param remove_dropout: Remove Dropout nodes (default True for inference)
    :param remove_redundant_operations: Remove no-op nodes
    :param reorder_by_strict_topological_order: Topological sort
    :param simplify_node_name: Rename nodes sequentially
    :param has_batch_dim: Whether model has batch dimension
    :return: Optimized ONNX model
    """
    graph_name = model.graph.name + "_slimmed"

    model = clear_onnx_docstring(model)

    # Convert constants to initializers (critical for shape inference and optimization)
    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)

    nodes = list(model.graph.node)
    nodes = _constant_to_initializer(nodes, initializers)

    # Update model with converted constants before any shape inference
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(list(initializers.values()))

    if remove_dropout:
        model = _remove_dropout(model)
        nodes = list(model.graph.node)

    # Apply shape-based optimizations
    nodes, initializers = _apply_shape_based_optimizations(
        nodes,
        initializers,
        input_nodes,
        output_nodes,
        has_batch_dim,
        constant_folding,
        remove_redundant_operations,
        simplify_conv_to_flatten_gemm,
    )

    # Apply Gemm and BatchNorm fusions
    nodes = _apply_gemm_bn_fusions(
        nodes,
        initializers,
        input_nodes,
        output_nodes,
        has_batch_dim,
        fuse_matmul_add,
        fuse_gemm_reshape_bn,
        fuse_bn_reshape_gemm,
        fuse_bn_gemm,
        fuse_transpose_bn_transpose,
        simplify_gemm,
        fuse_gemm_gemm,
    )

    # Apply Conv fusions
    nodes = _apply_conv_fusions(
        nodes,
        initializers,
        fuse_conv_bn,
        fuse_bn_conv,
        fuse_conv_transpose_bn,
        fuse_bn_conv_transpose,
        fuse_depthwise_conv_bn,
        fuse_bn_depthwise_conv,
    )

    # Apply final processing
    if reorder_by_strict_topological_order:
        nodes, initializers = _simplify_names(input_nodes, output_nodes, nodes, initializers)
        nodes = _reorder_by_strict_topological_order(nodes)
    if simplify_node_name:
        nodes, initializers = _simplify_names(input_nodes, output_nodes, nodes, initializers)

    # Set the opset version not too high to ensure compatibility
    new_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            graph_name,
            input_nodes,
            output_nodes,
            list(initializers.values()),
        ),
        opset_imports=model.opset_import,  # Hold original opset versions
    )

    return new_model
