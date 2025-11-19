"""Create small ONNX models with specific optimization patterns for testing.

This module programmatically creates synthetic ONNX models that contain
specific patterns detectable by SlimONNX pattern detection system.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "create_matmul_add_pattern",
    "create_conv_bn_pattern",
    "create_bn_conv_pattern",
    "create_convtranspose_bn_pattern",
    "create_bn_convtranspose_pattern",
    "create_depthwise_conv_bn_pattern",
    "create_gemm_reshape_bn_pattern",
    "create_bn_reshape_gemm_pattern",
    "create_bn_gemm_pattern",
    "create_transpose_bn_transpose_pattern",
    "create_gemm_gemm_pattern",
    "create_dropout_pattern",
    "create_redundant_ops_pattern",
    "create_constant_folding_pattern",
    "create_all_patterns",
]

import numpy as np
import onnx
from onnx import TensorProto, helper


def create_matmul_add_pattern(output_path: str) -> str:
    """Create model with MatMul + Add pattern (fusable to Gemm).

    Pattern: MatMul(X, W) + bias
    Can be fused to: Gemm(X, W, bias)

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    # Create initializers
    weight = np.random.randn(4, 3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)

    weight_tensor = helper.make_tensor("weight", TensorProto.FLOAT, [4, 3], weight)
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [3], bias)

    # Create nodes
    matmul_node = helper.make_node("MatMul", ["input", "weight"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "bias"], ["output"])

    # Create graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        "matmul_add_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [weight_tensor, bias_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_conv_bn_pattern(output_path: str) -> str:
    """Create model with Conv + BatchNormalization pattern.

    Pattern: Conv -> BatchNormalization
    Can be fused by folding BN into Conv weights/bias.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    # Conv parameters: [out_channels, in_channels, kH, kW]
    conv_weight = np.random.randn(8, 3, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(8).astype(np.float32)

    # BN parameters
    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    # Create initializers
    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 3, 3, 3], conv_weight),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [8], conv_bias),
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
    ]

    # Create nodes
    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_w", "conv_b"],
        ["conv_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["output"],
        epsilon=1e-5,
    )

    # Create graph
    graph = helper.make_graph(
        [conv_node, bn_node],
        "conv_bn_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 32, 32])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_bn_conv_pattern(output_path: str) -> str:
    """Create model with BatchNormalization + Conv pattern.

    Pattern: BatchNormalization -> Conv
    Can be fused by folding BN into Conv weights.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    # BN parameters
    bn_scale = np.random.randn(3).astype(np.float32)
    bn_bias = np.random.randn(3).astype(np.float32)
    bn_mean = np.random.randn(3).astype(np.float32)
    bn_var = np.abs(np.random.randn(3)).astype(np.float32) + 0.1

    # Conv parameters
    conv_weight = np.random.randn(8, 3, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(8).astype(np.float32)

    initializers = [
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [3], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [3], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [3], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [3], bn_var),
        helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 3, 3, 3], conv_weight),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [8], conv_bias),
    ]

    bn_node = helper.make_node(
        "BatchNormalization",
        ["input", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        epsilon=1e-5,
    )
    conv_node = helper.make_node(
        "Conv",
        ["bn_out", "conv_w", "conv_b"],
        ["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        [bn_node, conv_node],
        "bn_conv_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 32, 32])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_convtranspose_bn_pattern(output_path: str) -> str:
    """Create model with ConvTranspose + BatchNormalization pattern.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    # ConvTranspose weight: [in_channels, out_channels, kH, kW]
    conv_weight = np.random.randn(8, 16, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(16).astype(np.float32)

    bn_scale = np.random.randn(16).astype(np.float32)
    bn_bias = np.random.randn(16).astype(np.float32)
    bn_mean = np.random.randn(16).astype(np.float32)
    bn_var = np.abs(np.random.randn(16)).astype(np.float32) + 0.1

    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 16, 3, 3], conv_weight),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [16], conv_bias),
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [16], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [16], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [16], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [16], bn_var),
    ]

    conv_node = helper.make_node(
        "ConvTranspose",
        ["input", "conv_w", "conv_b"],
        ["conv_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["output"],
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [conv_node, bn_node],
        "convtranspose_bn_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 32, 32])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_bn_convtranspose_pattern(output_path: str) -> str:
    """Create model with BatchNormalization + ConvTranspose pattern.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    conv_weight = np.random.randn(8, 16, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(16).astype(np.float32)

    initializers = [
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
        helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 16, 3, 3], conv_weight),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [16], conv_bias),
    ]

    bn_node = helper.make_node(
        "BatchNormalization",
        ["input", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        epsilon=1e-5,
    )
    conv_node = helper.make_node(
        "ConvTranspose",
        ["bn_out", "conv_w", "conv_b"],
        ["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )

    graph = helper.make_graph(
        [bn_node, conv_node],
        "bn_convtranspose_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, 32, 32])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_depthwise_conv_bn_pattern(output_path: str) -> str:
    """Create model with Depthwise Conv + BatchNormalization pattern.

    Depthwise conv: group = in_channels = out_channels
    Weight shape: [C, 1, kH, kW]

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    channels = 8
    # Depthwise conv weight: [out_channels=8, in_channels_per_group=1, kH, kW]
    conv_weight = np.random.randn(channels, 1, 3, 3).astype(np.float32)
    conv_bias = np.random.randn(channels).astype(np.float32)

    bn_scale = np.random.randn(channels).astype(np.float32)
    bn_bias = np.random.randn(channels).astype(np.float32)
    bn_mean = np.random.randn(channels).astype(np.float32)
    bn_var = np.abs(np.random.randn(channels)).astype(np.float32) + 0.1

    initializers = [
        helper.make_tensor("conv_w", TensorProto.FLOAT, [8, 1, 3, 3], conv_weight),
        helper.make_tensor("conv_b", TensorProto.FLOAT, [8], conv_bias),
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
    ]

    # Depthwise conv: group = out_channels = in_channels
    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_w", "conv_b"],
        ["conv_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=8,  # Depthwise
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["output"],
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [conv_node, bn_node],
        "depthwise_conv_bn_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 32, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 32, 32])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_gemm_reshape_bn_pattern(output_path: str) -> str:
    """Create model with Gemm + Reshape + BatchNormalization pattern.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    gemm_weight = np.random.randn(10, 8).astype(np.float32)
    gemm_bias = np.random.randn(8).astype(np.float32)

    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    reshape_shape = np.array([1, 8, 1, 1], dtype=np.int64)

    initializers = [
        helper.make_tensor("gemm_w", TensorProto.FLOAT, [10, 8], gemm_weight),
        helper.make_tensor("gemm_b", TensorProto.FLOAT, [8], gemm_bias),
        helper.make_tensor("reshape_shape", TensorProto.INT64, [4], reshape_shape),
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
    ]

    gemm_node = helper.make_node(
        "Gemm", ["input", "gemm_w", "gemm_b"], ["gemm_out"], transB=1
    )
    reshape_node = helper.make_node(
        "Reshape", ["gemm_out", "reshape_shape"], ["reshape_out"]
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        ["reshape_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["output"],
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [gemm_node, reshape_node, bn_node],
        "gemm_reshape_bn_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 1, 1])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_bn_reshape_gemm_pattern(output_path: str) -> str:
    """Create model with BatchNormalization + Reshape + Gemm pattern.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    reshape_shape = np.array([1, 8], dtype=np.int64)

    gemm_weight = np.random.randn(8, 4).astype(np.float32)
    gemm_bias = np.random.randn(4).astype(np.float32)

    initializers = [
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
        helper.make_tensor("reshape_shape", TensorProto.INT64, [2], reshape_shape),
        helper.make_tensor("gemm_w", TensorProto.FLOAT, [8, 4], gemm_weight),
        helper.make_tensor("gemm_b", TensorProto.FLOAT, [4], gemm_bias),
    ]

    bn_node = helper.make_node(
        "BatchNormalization",
        ["input", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        epsilon=1e-5,
    )
    reshape_node = helper.make_node(
        "Reshape", ["bn_out", "reshape_shape"], ["reshape_out"]
    )
    gemm_node = helper.make_node(
        "Gemm", ["reshape_out", "gemm_w", "gemm_b"], ["output"], transB=1
    )

    graph = helper.make_graph(
        [bn_node, reshape_node, gemm_node],
        "bn_reshape_gemm_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 1, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_bn_gemm_pattern(output_path: str) -> str:
    """Create model with BatchNormalization + Gemm pattern.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    gemm_weight = np.random.randn(8, 4).astype(np.float32)
    gemm_bias = np.random.randn(4).astype(np.float32)

    initializers = [
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
        helper.make_tensor("gemm_w", TensorProto.FLOAT, [8, 4], gemm_weight),
        helper.make_tensor("gemm_b", TensorProto.FLOAT, [4], gemm_bias),
    ]

    bn_node = helper.make_node(
        "BatchNormalization",
        ["input", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        epsilon=1e-5,
    )
    gemm_node = helper.make_node("Gemm", ["bn_out", "gemm_w", "gemm_b"], ["output"])

    graph = helper.make_graph(
        [bn_node, gemm_node],
        "bn_gemm_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_transpose_bn_transpose_pattern(output_path: str) -> str:
    """Create model with Transpose + BatchNormalization + Transpose pattern.

    Pattern: Transpose(0,2,1) -> BN -> Transpose(0,2,1)
    Can be fused into single Gemm operation.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    bn_scale = np.random.randn(8).astype(np.float32)
    bn_bias = np.random.randn(8).astype(np.float32)
    bn_mean = np.random.randn(8).astype(np.float32)
    bn_var = np.abs(np.random.randn(8)).astype(np.float32) + 0.1

    initializers = [
        helper.make_tensor("bn_scale", TensorProto.FLOAT, [8], bn_scale),
        helper.make_tensor("bn_bias", TensorProto.FLOAT, [8], bn_bias),
        helper.make_tensor("bn_mean", TensorProto.FLOAT, [8], bn_mean),
        helper.make_tensor("bn_var", TensorProto.FLOAT, [8], bn_var),
    ]

    transpose1_node = helper.make_node(
        "Transpose", ["input"], ["transpose1_out"], perm=[0, 2, 1]
    )
    bn_node = helper.make_node(
        "BatchNormalization",
        ["transpose1_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["bn_out"],
        epsilon=1e-5,
    )
    transpose2_node = helper.make_node(
        "Transpose", ["bn_out"], ["output"], perm=[0, 2, 1]
    )

    graph = helper.make_graph(
        [transpose1_node, bn_node, transpose2_node],
        "transpose_bn_transpose_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_gemm_gemm_pattern(output_path: str) -> str:
    """Create model with consecutive Gemm operations (linear merging).

    Pattern: Gemm -> Gemm
    Can be merged: (X @ W1 + b1) @ W2 + b2 = X @ (W1 @ W2) + (b1 @ W2 + b2)

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    weight1 = np.random.randn(8, 6).astype(np.float32)
    bias1 = np.random.randn(6).astype(np.float32)

    weight2 = np.random.randn(6, 4).astype(np.float32)
    bias2 = np.random.randn(4).astype(np.float32)

    initializers = [
        helper.make_tensor("weight1", TensorProto.FLOAT, [8, 6], weight1),
        helper.make_tensor("bias1", TensorProto.FLOAT, [6], bias1),
        helper.make_tensor("weight2", TensorProto.FLOAT, [6, 4], weight2),
        helper.make_tensor("bias2", TensorProto.FLOAT, [4], bias2),
    ]

    gemm1_node = helper.make_node("Gemm", ["input", "weight1", "bias1"], ["gemm1_out"])
    gemm2_node = helper.make_node("Gemm", ["gemm1_out", "weight2", "bias2"], ["output"])

    graph = helper.make_graph(
        [gemm1_node, gemm2_node],
        "gemm_gemm_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_dropout_pattern(output_path: str) -> str:
    """Create model with Dropout nodes (should be removed for inference).

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    weight = np.random.randn(8, 4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)

    initializers = [
        helper.make_tensor("weight", TensorProto.FLOAT, [8, 4], weight),
        helper.make_tensor("bias", TensorProto.FLOAT, [4], bias),
    ]

    gemm_node = helper.make_node("Gemm", ["input", "weight", "bias"], ["gemm_out"])
    dropout_node = helper.make_node("Dropout", ["gemm_out"], ["dropout_out"], ratio=0.5)
    relu_node = helper.make_node("Relu", ["dropout_out"], ["output"])

    graph = helper.make_graph(
        [gemm_node, dropout_node, relu_node],
        "dropout_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_redundant_ops_pattern(output_path: str) -> str:
    """Create model with redundant operations (add zero, mul one, etc).

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    zero = np.zeros(4, dtype=np.float32)
    one = np.ones(4, dtype=np.float32)
    weight = np.random.randn(8, 4).astype(np.float32)

    initializers = [
        helper.make_tensor("zero", TensorProto.FLOAT, [4], zero),
        helper.make_tensor("one", TensorProto.FLOAT, [4], one),
        helper.make_tensor("weight", TensorProto.FLOAT, [8, 4], weight),
    ]

    matmul_node = helper.make_node("MatMul", ["input", "weight"], ["matmul_out"])
    add_zero_node = helper.make_node("Add", ["matmul_out", "zero"], ["add_out"])
    mul_one_node = helper.make_node("Mul", ["add_out", "one"], ["mul_out"])
    sub_zero_node = helper.make_node("Sub", ["mul_out", "zero"], ["sub_out"])
    div_one_node = helper.make_node("Div", ["sub_out", "one"], ["output"])

    graph = helper.make_graph(
        [matmul_node, add_zero_node, mul_one_node, sub_zero_node, div_one_node],
        "redundant_ops_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_constant_folding_pattern(output_path: str) -> str:
    """Create model with constant folding opportunities.

    Operations with all-constant inputs can be pre-computed.

    :param output_path: Path to save ONNX model
    :return: Path to saved model
    """
    const1 = np.array([2, 4, 6, 8], dtype=np.float32)
    const2 = np.array([1, 2, 3, 4], dtype=np.float32)
    weight = np.random.randn(4, 4).astype(np.float32)

    initializers = [
        helper.make_tensor("const1", TensorProto.FLOAT, [4], const1),
        helper.make_tensor("const2", TensorProto.FLOAT, [4], const2),
        helper.make_tensor("weight", TensorProto.FLOAT, [4, 4], weight),
    ]

    # Constant operations (can be folded)
    add_const_node = helper.make_node("Add", ["const1", "const2"], ["const_sum"])
    mul_const_node = helper.make_node("Mul", ["const_sum", "const2"], ["const_product"])

    # Non-constant operation
    add_node = helper.make_node("Add", ["input", "const_product"], ["add_out"])
    matmul_node = helper.make_node("MatMul", ["add_out", "weight"], ["output"])

    graph = helper.make_graph(
        [add_const_node, mul_const_node, add_node, matmul_node],
        "constant_folding_pattern",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)
    return output_path


def create_all_patterns(output_dir: str) -> dict[str, str]:
    """Create all pattern test models in the specified directory.

    :param output_dir: Directory to save all pattern models
    :return: Dictionary mapping pattern names to file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    patterns = {
        "matmul_add": create_matmul_add_pattern,
        "conv_bn": create_conv_bn_pattern,
        "bn_conv": create_bn_conv_pattern,
        "convtranspose_bn": create_convtranspose_bn_pattern,
        "bn_convtranspose": create_bn_convtranspose_pattern,
        "depthwise_conv_bn": create_depthwise_conv_bn_pattern,
        "gemm_reshape_bn": create_gemm_reshape_bn_pattern,
        "bn_reshape_gemm": create_bn_reshape_gemm_pattern,
        "bn_gemm": create_bn_gemm_pattern,
        "transpose_bn_transpose": create_transpose_bn_transpose_pattern,
        "gemm_gemm": create_gemm_gemm_pattern,
        "dropout": create_dropout_pattern,
        "redundant_ops": create_redundant_ops_pattern,
        "constant_folding": create_constant_folding_pattern,
    }

    created_files = {}
    for pattern_name, create_func in patterns.items():
        output_path = os.path.join(output_dir, f"{pattern_name}.onnx")
        create_func(output_path)
        created_files[pattern_name] = output_path
        print(f"Created: {output_path}")

    return created_files


if __name__ == "__main__":
    import sys
    import os

    # Add parent directories to path
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    output_dir = os.path.join(os.path.dirname(__file__), "models")
    print(f"Creating pattern test models in: {output_dir}")
    print("=" * 80)

    created_files = create_all_patterns(output_dir)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Created {len(created_files)} pattern test models")
    print("=" * 80)
