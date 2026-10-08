"""Remove Dropout nodes for inference optimization."""

__docformat__ = "restructuredtext"
__all__ = ["remove_dropout"]

from onnx import ModelProto, NodeProto, TensorProto, numpy_helper


def _is_static_inference_mode(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    *,
    default_inference_mode: bool,
) -> bool:
    """Return whether a Dropout node is statically in inference mode."""
    legacy_is_test = next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "is_test"),
        None,
    )
    if legacy_is_test is not None:
        return legacy_is_test == 1

    if len(node.input) < 3 or not node.input[2]:
        return default_inference_mode

    training_mode = initializers.get(node.input[2])
    if training_mode is None or training_mode.data_type != TensorProto.BOOL:
        return False

    value = numpy_helper.to_array(training_mode)
    return bool(value.shape == () and not value.item())


def _build_dropout_mapping(
    nodes: list[NodeProto],
    initializers: dict[str, TensorProto] | None = None,
    graph_output_names: set[str] | None = None,
    *,
    default_inference_mode: bool = True,
) -> tuple[dict[str, str], list[NodeProto]]:
    """Build a mapping for Dropout nodes proved safe to bypass.

    :param nodes: All model nodes.

    :param initializers: Model initializer map used to resolve a static
        ``training_mode`` input.

    :param graph_output_names: Observable graph output value names.

    :param default_inference_mode: Semantics when neither a legacy
        ``is_test`` attribute nor a modern ``training_mode`` input is present.

    :return: Tuple of (dropout_output_to_input mapping, nodes_to_remove list)
    """
    initializers = initializers or {}
    graph_output_names = graph_output_names or set()
    consumed_values = {name for node in nodes for name in node.input if name}
    observable_values = consumed_values | graph_output_names
    dropout_output_to_input: dict[str, str] = {}
    nodes_to_remove: list[NodeProto] = []

    for node in nodes:
        if (
            node.op_type != "Dropout"
            or node.domain not in {"", "ai.onnx"}
            or not node.input
            or not node.input[0]
            or not node.output
            or len(node.output) > 2
            or not node.output[0]
            or node.output[0] in graph_output_names
            or not _is_static_inference_mode(
                node,
                initializers,
                default_inference_mode=default_inference_mode,
            )
        ):
            continue

        mask_output = node.output[1] if len(node.output) > 1 else ""
        if mask_output and mask_output in observable_values:
            continue

        dropout_output_to_input[node.output[0]] = node.input[0]
        nodes_to_remove.append(node)

    return dropout_output_to_input, nodes_to_remove


def _resolve_dropout_source(value_name: str, dropout_mapping: dict[str, str]) -> str:
    """Resolve a value through a chain of bypassed Dropout nodes."""
    source = value_name
    visited: set[str] = set()
    while source in dropout_mapping:
        if source in visited:
            raise ValueError(f"Dropout mapping contains a cycle at {source!r}")
        visited.add(source)
        source = dropout_mapping[source]
    return source


def _update_node_inputs(
    nodes: list[NodeProto],
    nodes_to_remove: list[NodeProto],
    dropout_mapping: dict[str, str],
) -> list[NodeProto]:
    """Update node inputs to bypass dropout nodes.

    :param nodes: All model nodes.

    :param nodes_to_remove: Dropout nodes to remove.

    :param dropout_mapping: Mapping of dropout outputs to inputs.

    :return: New list of nodes with updated inputs
    """
    removed_ids = {id(node) for node in nodes_to_remove}
    new_nodes: list[NodeProto] = []
    for node in nodes:
        if id(node) in removed_ids:
            continue

        new_input = [_resolve_dropout_source(name, dropout_mapping) for name in node.input]
        modified = new_input != list(node.input)

        if modified:
            new_node = NodeProto()
            new_node.CopyFrom(node)
            new_node.ClearField("input")
            new_node.input.extend(new_input)
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    return new_nodes


def remove_dropout(model: ModelProto) -> ModelProto:
    """Remove Dropout nodes proved to be unobservable inference no-ops.

    Dropout is a training-only operation that randomly zeros elements during training.
    During inference, Dropout nodes are redundant and should be removed.

    A node is bypassed only when ``training_mode`` is absent or a scalar
    ``False`` initializer and its optional mask output is not consumed or
    exposed as a graph output. A node whose primary output is public is kept
    rather than changing that public value name. Consecutive removable
    Dropout chains are resolved to their transitive source.

    :param model: ONNX model.

    :return: Optimized model with Dropout nodes removed
    """
    graph = model.graph
    nodes = list(graph.node)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    graph_output_names = {output.name for output in graph.output}
    default_domain_opset = next(
        (opset.version for opset in model.opset_import if opset.domain in {"", "ai.onnx"}),
        None,
    )
    default_inference_mode = default_domain_opset is not None and default_domain_opset >= 7

    dropout_mapping, nodes_to_remove = _build_dropout_mapping(
        nodes,
        initializers,
        graph_output_names,
        default_inference_mode=default_inference_mode,
    )

    if not nodes_to_remove:
        return model

    new_nodes = _update_node_inputs(nodes, nodes_to_remove, dropout_mapping)

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    return model
