"""Main optimization orchestration for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = ["optimize_onnx"]

from typing import Any

import onnx
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto
from shapeonnx.infer_shape import infer_onnx_shape

from slimonnx.configs import OptimizationConfig
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
from slimonnx.optimize_onnx._cst_op import _constant_to_initializer, _fuse_constant_nodes
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


def _infer_shapes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    has_batch_dim: bool,
) -> Any:
    """Run shapeonnx shape inference on the current graph state.

    Return type is ``Any`` because ``shapeonnx.infer_onnx_shape`` is
    untyped; downstream pass helpers accept whatever shape map shapeonnx
    produces.
    """
    return infer_onnx_shape(input_nodes, output_nodes, nodes, initializers, has_batch_dim)


def _run_shape_based_passes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    config: OptimizationConfig,
) -> tuple[list[NodeProto], dict[str, TensorProto]]:
    """Run optimizations that consume shape inference output.

    Shape inference is cached between passes via a ``None`` sentinel: after
    any mutation to ``nodes`` or ``initializers`` the sentinel forces a
    recompute before the next shape-dependent pass.  This avoids redundant
    ``O(nodes)`` shape inference when a sub-pass is skipped by config or
    produces no changes.
    """
    has_batch_dim = config.has_batch_dim
    data_shapes: Any = None

    if config.constant_folding:
        data_shapes = _infer_shapes(nodes, initializers, input_nodes, output_nodes, has_batch_dim)
        nodes, initializers = _fuse_constant_nodes(nodes, initializers, data_shapes)
        data_shapes = None  # invalidated by mutation above

    if data_shapes is None:
        data_shapes = _infer_shapes(nodes, initializers, input_nodes, output_nodes, has_batch_dim)
    nodes = _resolve_reshape_negative_one(nodes, initializers, data_shapes)
    data_shapes = None  # _resolve_reshape may have changed shapes

    if config.remove_redundant_operations:
        if data_shapes is None:
            data_shapes = _infer_shapes(
                nodes, initializers, input_nodes, output_nodes, has_batch_dim
            )
        nodes = _reorder_by_strict_topological_order(nodes)
        nodes = _remove_redundant_operations(nodes, initializers, data_shapes, output_nodes)
        data_shapes = None

    if config.simplify_conv_to_flatten_gemm:
        if data_shapes is None:
            data_shapes = _infer_shapes(
                nodes, initializers, input_nodes, output_nodes, has_batch_dim
            )
        nodes = _simplify_conv_to_flatten_gemm(nodes, initializers, data_shapes)

    return nodes, initializers


def _run_gemm_bn_passes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    input_nodes: list[ValueInfoProto],
    output_nodes: list[ValueInfoProto],
    config: OptimizationConfig,
    simplify_gemm: bool,
) -> list[NodeProto]:
    """Run Gemm + BatchNorm fusion passes."""
    has_batch_dim = config.has_batch_dim
    data_shapes: Any = None

    if config.fuse_matmul_add:
        data_shapes = _infer_shapes(nodes, initializers, input_nodes, output_nodes, has_batch_dim)
        nodes = _fuse_matmul_add(nodes, initializers, input_nodes, data_shapes)
        data_shapes = None

    if config.fuse_gemm_reshape_bn:
        nodes = _fuse_gemm_reshape_bn(nodes, initializers)
        data_shapes = None
    if config.fuse_bn_reshape_gemm:
        nodes = _fuse_bn_reshape_gemm(nodes, initializers)
        data_shapes = None
    if config.fuse_bn_gemm:
        nodes = _fuse_bn_gemm(nodes, initializers)
        data_shapes = None

    if config.fuse_transpose_bn_transpose:
        if data_shapes is None:
            data_shapes = _infer_shapes(
                nodes, initializers, input_nodes, output_nodes, has_batch_dim
            )
        nodes = _fuse_transpose_batchnorm_transpose(nodes, initializers, input_nodes, data_shapes)
        data_shapes = None

    if simplify_gemm:
        nodes = _simplify_gemm(nodes, initializers)

    if config.fuse_gemm_gemm:
        # Iterate to a fixpoint: each pass can expose new Gemm-Gemm chains
        # by folding earlier ones. Bounded by node count to guarantee
        # termination if a pass ever fails to reduce node count.
        max_iters = max(1, len(nodes))
        for _ in range(max_iters):
            before = len(nodes)
            nodes = _fuse_gemm_gemm(nodes, initializers)
            if len(nodes) == before:
                break

    return nodes


def _run_conv_passes(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto],
    config: OptimizationConfig,
) -> list[NodeProto]:
    """Run Conv + BatchNorm fusion passes."""
    if config.fuse_conv_bn:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers)
    if config.fuse_bn_conv:
        nodes = _fuse_conv_bn_or_bn_conv(nodes, initializers, is_conv_bn=False)

    if config.fuse_conv_transpose_bn:
        nodes = _fuse_conv_transpose_bn_or_bn_conv_transpose(nodes, initializers)
    if config.fuse_bn_conv_transpose:
        nodes = _fuse_conv_transpose_bn_or_bn_conv_transpose(
            nodes, initializers, is_conv_transpose_bn=False
        )

    if config.fuse_depthwise_conv_bn:
        nodes = _fuse_depthwise_conv_bn_or_bn_depthwise_conv(nodes, initializers)
    if config.fuse_bn_depthwise_conv:
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

    :param model: Input ONNX model.

    :param constant_folding: Fold constant operations (renamed from fuse_constant_nodes).

    :param fuse_matmul_add: Fuse MatMul + Add to Gemm.

    :param fuse_gemm_reshape_bn: Fuse Gemm-Reshape-BatchNorm.

    :param fuse_bn_reshape_gemm: Fuse BatchNorm-Reshape-Gemm.

    :param fuse_bn_gemm: Fuse BatchNorm-Gemm.

    :param fuse_transpose_bn_transpose: Fuse Transpose-BN-Transpose.

    :param fuse_gemm_gemm: Fuse consecutive Gemm nodes.

    :param fuse_conv_bn: Fuse Conv-BatchNorm.

    :param fuse_bn_conv: Fuse BatchNorm-Conv (skips cases with padding).

    :param fuse_conv_transpose_bn: Fuse ConvTranspose-BatchNorm.

    :param fuse_bn_conv_transpose: Fuse BatchNorm-ConvTranspose.

    :param fuse_depthwise_conv_bn: Fuse Depthwise Conv-BatchNorm.

    :param fuse_bn_depthwise_conv: Fuse BatchNorm-Depthwise Conv.

    :param simplify_conv_to_flatten_gemm: Convert Conv to Flatten-Gemm.

    :param simplify_gemm: Normalize Gemm attributes.

    :param remove_dropout: Remove Dropout nodes (default True for inference).

    :param remove_redundant_operations: Remove no-op nodes.

    :param reorder_by_strict_topological_order: Topological sort.

    :param simplify_node_name: Rename nodes sequentially.

    :param has_batch_dim: Whether model has batch dimension.

    :return: Optimized ONNX model
    """
    # Bridge legacy bool kwargs to the internal OptimizationConfig so the pass
    # drivers can take a single, immutable config object.
    config = OptimizationConfig(
        constant_folding=constant_folding,
        fuse_matmul_add=fuse_matmul_add,
        fuse_gemm_reshape_bn=fuse_gemm_reshape_bn,
        fuse_bn_reshape_gemm=fuse_bn_reshape_gemm,
        fuse_bn_gemm=fuse_bn_gemm,
        fuse_transpose_bn_transpose=fuse_transpose_bn_transpose,
        fuse_gemm_gemm=fuse_gemm_gemm,
        fuse_conv_bn=fuse_conv_bn,
        fuse_bn_conv=fuse_bn_conv,
        fuse_conv_transpose_bn=fuse_conv_transpose_bn,
        fuse_bn_conv_transpose=fuse_bn_conv_transpose,
        fuse_depthwise_conv_bn=fuse_depthwise_conv_bn,
        fuse_bn_depthwise_conv=fuse_bn_depthwise_conv,
        simplify_conv_to_flatten_gemm=simplify_conv_to_flatten_gemm,
        remove_redundant_operations=remove_redundant_operations,
        remove_dropout=remove_dropout,
        simplify_node_name=simplify_node_name,
        has_batch_dim=has_batch_dim,
    )
    return _optimize_with_config(
        model,
        config,
        simplify_gemm=simplify_gemm,
        reorder_by_strict_topological_order=reorder_by_strict_topological_order,
    )


def _optimize_with_config(
    model: ModelProto,
    config: OptimizationConfig,
    *,
    simplify_gemm: bool,
    reorder_by_strict_topological_order: bool,
) -> ModelProto:
    """Drive the optimization pipeline from an OptimizationConfig.

    Internal entry point used by both ``optimize_onnx`` (legacy kwarg facade)
    and the SlimONNX toolkit. ``simplify_gemm`` and
    ``reorder_by_strict_topological_order`` are kept as separate kwargs because
    they are pipeline invariants at the SlimONNX level but opt-in here so unit
    tests can isolate them.
    """
    has_batch_dim = config.has_batch_dim

    graph_name = model.graph.name + "_slimmed"
    model = clear_onnx_docstring(model)

    initializers = get_initializers(model)
    input_nodes = get_input_nodes(model, initializers, has_batch_dim)
    output_nodes = get_output_nodes(model, has_batch_dim)

    nodes = list(model.graph.node)
    nodes = _constant_to_initializer(nodes, initializers)

    # Push the post-folding graph back onto ``model`` before shape inference:
    # shapeonnx reads from input ValueInfoProto, but the Constant->initializer
    # move is required for inference on graphs with Constant-bound shapes.
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(list(initializers.values()))

    if config.remove_dropout:
        model = _remove_dropout(model)
        nodes = list(model.graph.node)
        output_nodes = get_output_nodes(model, has_batch_dim)

    nodes, initializers = _run_shape_based_passes(
        nodes, initializers, input_nodes, output_nodes, config
    )
    nodes = _run_gemm_bn_passes(
        nodes, initializers, input_nodes, output_nodes, config, simplify_gemm
    )
    nodes = _run_conv_passes(nodes, initializers, config)

    if reorder_by_strict_topological_order:
        nodes, initializers = _simplify_names(input_nodes, output_nodes, nodes, initializers)
        nodes = _reorder_by_strict_topological_order(nodes)
    if config.simplify_node_name:
        nodes, initializers = _simplify_names(input_nodes, output_nodes, nodes, initializers)

    new_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            graph_name,
            input_nodes,
            output_nodes,
            list(initializers.values()),
        ),
        opset_imports=model.opset_import,
    )
    return new_model
