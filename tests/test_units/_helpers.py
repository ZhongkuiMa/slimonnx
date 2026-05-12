"""Shared ONNX model builder helpers for slimonnx unit tests."""

__docformat__ = "restructuredtext"

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def create_tensor_value_info(name, dtype, shape):
    """Create a tensor value info for ONNX graph."""
    if dtype == "float32":
        onnx_dtype = TensorProto.FLOAT
    elif dtype == "int64":
        onnx_dtype = TensorProto.INT64
    elif dtype == "int32":
        onnx_dtype = TensorProto.INT32
    else:
        onnx_dtype = TensorProto.FLOAT

    return helper.make_tensor_value_info(name, onnx_dtype, shape)


def create_initializer(name, values, dtype="float32"):
    """Create an initializer (constant tensor) for ONNX graph."""
    if isinstance(values, np.ndarray):
        array = values
    else:
        array = np.array(values, dtype=dtype)

    if dtype == "float32":
        array = array.astype(np.float32)
    elif dtype == "int64":
        array = array.astype(np.int64)
    elif dtype == "int32":
        array = array.astype(np.int32)

    return numpy_helper.from_array(array, name=name)


def create_minimal_onnx_model(nodes, inputs, outputs, initializers=None):
    """Create minimal ONNX model for testing without file I/O.

    Args:
        nodes: List of ONNX node objects
        inputs: List of tensor value info objects (graph inputs)
        outputs: List of tensor value info objects (graph outputs)
        initializers: Optional list of initializer tensors

    Returns:
        ONNX ModelProto
    """
    graph = helper.make_graph(nodes, "test_graph", inputs, outputs, initializer=initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def count_ops_by_type(model, op_type):
    """Count nodes of specific op_type in model."""
    count = 0
    for node in model.graph.node:
        if node.op_type == op_type:
            count += 1
    return count


def get_nodes_by_type(model, op_type):
    """Get all nodes of specific op_type in model."""
    return [node for node in model.graph.node if node.op_type == op_type]


def run_onnx_model(model, inputs_dict):
    """Run ONNX model and return outputs.

    Args:
        model: ONNX ModelProto
        inputs_dict: Dict mapping input names to numpy arrays

    Returns:
        List of output arrays or dict of output name -> array
    """
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        return sess.run(None, inputs_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to run model: {e}") from e


def get_input_names(model):
    """Get list of input tensor names from model."""
    return [inp.name for inp in model.graph.input]


def get_initializer_names(model):
    """Get set of initializer names from model."""
    return {init.name for init in model.graph.initializer}


def get_node_by_name(model, node_name):
    """Get node by name from model, or None if not found."""
    for node in model.graph.node:
        if node.name == node_name:
            return node
    return None


def get_initializer_by_name(model, init_name):
    """Get initializer by name from model, or None if not found."""
    for init in model.graph.initializer:
        if init.name == init_name:
            return numpy_helper.to_array(init)
    return None
