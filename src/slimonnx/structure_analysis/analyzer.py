"""ONNX model structure analyzer."""

__docformat__ = "restructuredtext"
__all__ = ["analyze_inputs_outputs", "analyze_structure", "count_op_types"]

from collections import Counter

from onnx import ModelProto, NodeProto


def count_op_types(nodes: list[NodeProto]) -> dict[str, int]:
    """Count operations by type.

    :param nodes: Model nodes
    :return: Dictionary mapping op_type to count
    """
    return dict(Counter(node.op_type for node in nodes))


def analyze_inputs_outputs(model: ModelProto) -> dict:
    """Analyze model inputs and outputs.

    :param model: ONNX model
    :return: Input/output metadata dictionary
    """
    inputs_info = []
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.HasField("dim_value") else "?" for d in inp.type.tensor_type.shape.dim
        ]
        inputs_info.append(
            {
                "name": inp.name,
                "dtype": inp.type.tensor_type.elem_type,
                "shape": shape,
            }
        )

    outputs_info = []
    for out in model.graph.output:
        shape = [
            d.dim_value if d.HasField("dim_value") else "?" for d in out.type.tensor_type.shape.dim
        ]
        outputs_info.append(
            {
                "name": out.name,
                "dtype": out.type.tensor_type.elem_type,
                "shape": shape,
            }
        )

    return {
        "inputs": inputs_info,
        "outputs": outputs_info,
        "num_inputs": len(inputs_info),
        "num_outputs": len(outputs_info),
    }


def analyze_structure(
    model: ModelProto,
    data_shapes: dict[str, list[int]] | None = None,
) -> dict:
    """Analyze ONNX model structure.

    :param model: ONNX model
    :param data_shapes: Inferred shapes dictionary
    :return: Structure analysis results
    """
    nodes = list(model.graph.node)
    op_type_counts = count_op_types(nodes)
    io_info = analyze_inputs_outputs(model)

    from slimonnx import utils

    initializers = utils.get_initializers(model)

    return {
        "node_count": len(nodes),
        "op_type_counts": op_type_counts,
        "initializer_count": len(initializers),
        "inputs": io_info["inputs"],
        "outputs": io_info["outputs"],
        "num_inputs": io_info["num_inputs"],
        "num_outputs": io_info["num_outputs"],
        "has_shapes": data_shapes is not None,
    }
