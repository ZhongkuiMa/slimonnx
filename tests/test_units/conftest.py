"""Shared fixtures and utilities for unit tests."""

import numpy as np
import onnxruntime as ort
import pytest
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


@pytest.fixture
def simple_2x3_matrix():
    """Create a simple 2x3 matrix for testing."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def simple_3x2_matrix():
    """Create a simple 3x2 matrix for testing."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def simple_conv_input():
    """Create a simple 1x3x4x4 input for Conv testing (batch=1, C=3, H=4, W=4)."""
    return np.ones((1, 3, 4, 4), dtype=np.float32)


@pytest.fixture
def simple_float_model():
    """Create a simple model with float32 inputs and outputs."""
    inputs = [create_tensor_value_info("X", "float32", [2, 3])]
    outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

    # Identity node: Y = X
    nodes = [helper.make_node("Identity", inputs=["X"], outputs=["Y"])]

    return create_minimal_onnx_model(nodes, inputs, outputs)


@pytest.fixture
def onnx_model_runner():
    """Fixture to run ONNX models with onnxruntime."""

    def _run(model, input_dict):
        """Run model and return outputs."""
        return run_onnx_model(model, input_dict)

    return _run


@pytest.fixture
def create_matmul_add_model():
    """Create MatMul+Add model fixture."""

    def _create(shape_m=2, shape_n=3, shape_k=3, add_constant_bias=True):
        """Create MatMul+Add model.

        Args:
            shape_m: M dimension for MatMul
            shape_n: N dimension for MatMul
            shape_k: K dimension for MatMul
            add_constant_bias: If True, add is with constant; else variable
        """
        # Inputs
        X = create_tensor_value_info("X", "float32", [shape_m, shape_k])
        inputs = [X]

        # Initializers
        W = np.random.randn(shape_k, shape_n).astype(np.float32)
        if add_constant_bias:
            b = np.random.randn(shape_n).astype(np.float32)
            initializers = [
                create_initializer("W", W),
                create_initializer("b", b),
            ]
        else:
            initializers = [create_initializer("W", W)]
            inputs.append(create_tensor_value_info("b", "float32", [shape_n]))

        # Nodes
        matmul_node = helper.make_node("MatMul", inputs=["X", "W"], outputs=["matmul_output"])
        add_node = helper.make_node("Add", inputs=["matmul_output", "b"], outputs=["Y"])
        nodes = [matmul_node, add_node]

        # Outputs
        outputs = [create_tensor_value_info("Y", "float32", [shape_m, shape_n])]

        return create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    return _create


@pytest.fixture
def create_gemm_model():
    """Create Gemm model fixture."""

    def _create(alpha=1.0, beta=1.0, trans_a=0, trans_b=0, add_bias=True):
        """Create Gemm model.

        Args:
            alpha: Alpha attribute
            beta: Beta attribute
            trans_a: Transpose A attribute
            trans_b: Transpose B attribute
            add_bias: If True, add 3rd input (bias)
        """
        # Input shapes depend on transpose
        shape_A = [2, 3] if trans_a == 0 else [3, 2]
        shape_B = [3, 2] if trans_b == 0 else [2, 3]

        # Inputs
        A = create_tensor_value_info("A", "float32", shape_A)
        inputs = [A]

        # Initializers
        B_data = np.random.randn(*shape_B).astype(np.float32)
        initializers = [create_initializer("B", B_data)]

        if add_bias:
            C_data = np.random.randn(2).astype(np.float32)
            initializers.append(create_initializer("C", C_data))

        # Nodes
        gemm_inputs = ["A", "B", "C"] if add_bias else ["A", "B"]
        gemm_node = helper.make_node(
            "Gemm",
            inputs=gemm_inputs,
            outputs=["Y"],
            alpha=alpha,
            beta=beta,
            transA=trans_a,
            transB=trans_b,
        )
        nodes = [gemm_node]

        # Outputs
        outputs = [create_tensor_value_info("Y", "float32", [2, 2])]

        return create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    return _create


@pytest.fixture
def create_conv_bn_model():
    """Create Conv+BN model fixture."""

    def _create(in_channels=3, out_channels=2, kernel=3, input_hw=4):
        """Create Conv+BN model.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel: Kernel size
            input_hw: Input height/width
        """
        # Calculate output spatial dimensions
        output_hw = input_hw - kernel + 1

        # Inputs
        X = create_tensor_value_info("X", "float32", [1, in_channels, input_hw, input_hw])
        inputs = [X]

        # Conv initializers
        conv_w = np.random.randn(out_channels, in_channels, kernel, kernel).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        # BN initializers
        bn_scale = np.ones(out_channels, dtype=np.float32)
        bn_bias = np.zeros(out_channels, dtype=np.float32)
        bn_mean = np.zeros(out_channels, dtype=np.float32)
        bn_var = np.ones(out_channels, dtype=np.float32)

        initializers.extend(
            [
                create_initializer("bn_scale", bn_scale),
                create_initializer("bn_bias", bn_bias),
                create_initializer("bn_mean", bn_mean),
                create_initializer("bn_var", bn_var),
            ]
        )

        # Nodes
        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["conv_output"],
            kernel_shape=[kernel, kernel],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_output", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )
        nodes = [conv_node, bn_node]

        # Outputs
        outputs = [
            create_tensor_value_info("Y", "float32", [1, out_channels, output_hw, output_hw])
        ]

        return create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    return _create


@pytest.fixture
def create_dropout_model():
    """Create Dropout model fixture."""

    def _create(dropout_ratio=0.5):
        """Create Dropout model.

        Args:
            dropout_ratio: Dropout ratio
        """
        # Inputs
        X = create_tensor_value_info("X", "float32", [2, 3])
        inputs = [X]

        # Initializers - Dropout takes ratio as input in opset 12+ (must be scalar 0-D tensor)
        ratio_initializer = create_initializer("ratio", np.array(dropout_ratio, dtype=np.float32))

        # Nodes - Dropout takes X and ratio as inputs
        dropout_node = helper.make_node(
            "Dropout",
            inputs=["X", "ratio"],
            outputs=["Y"],
        )
        nodes = [dropout_node]

        # Outputs
        outputs = [create_tensor_value_info("Y", "float32", [2, 3])]

        return create_minimal_onnx_model(nodes, inputs, outputs, [ratio_initializer])

    return _create


@pytest.fixture
def create_convtranspose_bn_model():
    """Create ConvTranspose+BN model fixture."""

    def _create(in_channels=2, out_channels=3, kernel=3):
        """Create ConvTranspose+BN model.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel: Kernel size
        """
        input_hw = 4
        output_hw = input_hw + kernel - 1

        # Inputs
        X = create_tensor_value_info("X", "float32", [1, in_channels, input_hw, input_hw])
        inputs = [X]

        # ConvTranspose initializers
        conv_w = np.random.randn(in_channels, out_channels, kernel, kernel).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        # BN initializers
        bn_scale = np.ones(out_channels, dtype=np.float32)
        bn_bias = np.zeros(out_channels, dtype=np.float32)
        bn_mean = np.zeros(out_channels, dtype=np.float32)
        bn_var = np.ones(out_channels, dtype=np.float32)

        initializers.extend(
            [
                create_initializer("bn_scale", bn_scale),
                create_initializer("bn_bias", bn_bias),
                create_initializer("bn_mean", bn_mean),
                create_initializer("bn_var", bn_var),
            ]
        )

        # Nodes
        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["X", "conv_w"],
            outputs=["conv_output"],
            kernel_shape=[kernel, kernel],
        )
        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=["conv_output", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["Y"],
            epsilon=1e-5,
        )
        nodes = [conv_node, bn_node]

        # Outputs
        outputs = [
            create_tensor_value_info("Y", "float32", [1, out_channels, output_hw, output_hw])
        ]

        return create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    return _create


@pytest.fixture
def create_depthwise_conv_model():
    """Create depthwise Conv model fixture."""

    def _create(channels=2, kernel=3, input_hw=4):
        """Create depthwise Conv model.

        Args:
            channels: Number of channels (equal for depthwise)
            kernel: Kernel size
            input_hw: Input height/width
        """
        output_hw = input_hw - kernel + 1

        # Inputs
        X = create_tensor_value_info("X", "float32", [1, channels, input_hw, input_hw])
        inputs = [X]

        # Conv initializers (depthwise: out_channels=in_channels, group=channels)
        conv_w = np.random.randn(channels, 1, kernel, kernel).astype(np.float32)
        initializers = [create_initializer("conv_w", conv_w)]

        # Nodes
        conv_node = helper.make_node(
            "Conv",
            inputs=["X", "conv_w"],
            outputs=["Y"],
            kernel_shape=[kernel, kernel],
            group=channels,  # Depthwise: group == in_channels
        )
        nodes = [conv_node]

        # Outputs
        outputs = [create_tensor_value_info("Y", "float32", [1, channels, output_hw, output_hw])]

        return create_minimal_onnx_model(nodes, inputs, outputs, initializers)

    return _create
