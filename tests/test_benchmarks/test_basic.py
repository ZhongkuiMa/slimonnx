"""Basic functionality test for SlimONNX optimizations."""

__docformat__ = "restructuredtext"
__all__ = ["create_test_model", "test_basic_optimization", "test_conv_bn_fusion"]

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from slimonnx import OptimizationConfig
from slimonnx.slimonnx import SlimONNX

# Test tolerance constants
NUMERICAL_RTOL = 1e-5
NUMERICAL_ATOL = 1e-6


def create_test_model() -> onnx.ModelProto:
    """Create a simple ONNX model for testing Conv-BN fusion.

    Model structure: Input -> Conv -> BatchNorm -> ReLU -> Output

    :return: ONNX ModelProto for testing
    """
    input_tensor = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )

    output_tensor = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 64, 112, 112]
    )

    rng = np.random.default_rng()
    conv_w = rng.standard_normal((64, 3, 7, 7)).astype(np.float32)
    conv_w_init = onnx.numpy_helper.from_array(conv_w, name="conv_w")

    bn_scale = rng.standard_normal(64).astype(np.float32) + 1.0
    bn_bias = rng.standard_normal(64).astype(np.float32)
    bn_mean = rng.standard_normal(64).astype(np.float32)
    bn_var = np.abs(rng.standard_normal(64).astype(np.float32)) + 0.1

    bn_scale_init = onnx.numpy_helper.from_array(bn_scale, name="bn_scale")
    bn_bias_init = onnx.numpy_helper.from_array(bn_bias, name="bn_bias")
    bn_mean_init = onnx.numpy_helper.from_array(bn_mean, name="bn_mean")
    bn_var_init = onnx.numpy_helper.from_array(bn_var, name="bn_var")

    conv = onnx.helper.make_node(
        "Conv",
        inputs=["input", "conv_w"],
        outputs=["conv_out"],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    )

    bn = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        outputs=["bn_out"],
        epsilon=1e-5,
    )

    relu = onnx.helper.make_node("Relu", inputs=["bn_out"], outputs=["output"])

    graph = onnx.helper.make_graph(
        [conv, bn, relu],
        "test_conv_bn_model",
        [input_tensor],
        [output_tensor],
        [conv_w_init, bn_scale_init, bn_bias_init, bn_mean_init, bn_var_init],
    )

    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
    )


def _get_model_input_name(model_path: str) -> str:
    """Get first non-initializer input name from ONNX model.

    :param model_path: Path to ONNX model
    :return: Input name
    """
    model = onnx.load(model_path)
    return str(
        next(
            inp.name
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        )
    )


def _run_model(model_path: str, inputs: dict) -> dict:
    """Run ONNX model and return outputs.

    :param model_path: Path to ONNX model file
    :param inputs: Dictionary of input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, inputs)
    output_names = [out.name for out in session.get_outputs()]
    return dict(zip(output_names, outputs, strict=False))


def _prepare_optimized_model(model_path: str) -> None:
    """Convert model to opset 20 and IR version 10 for compatibility.

    :param model_path: Path to ONNX model to modify in-place
    """
    from slimonnx.preprocess import convert_model_version

    model = onnx.load(model_path)
    model = convert_model_version(model, target_opset=20, warn_on_diff=False)
    model.ir_version = 4
    onnx.save(model, model_path)


def test_basic_optimization() -> None:
    """Test basic SlimONNX optimization without fusion."""
    print("Creating test model...")
    model = create_test_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        original_path = str(tmpdir_path / "original.onnx")
        optimized_path = str(tmpdir_path / "optimized.onnx")

        onnx.save(model, original_path)
        print(f"Saved original model: {original_path}")

        print("Running SlimONNX optimization...")
        config = OptimizationConfig(
            has_batch_dim=True,
            fuse_conv_bn=False,
            simplify_node_name=True,
        )
        slimonnx = SlimONNX()
        slimonnx.slim(original_path, optimized_path, config=config)

        assert Path(optimized_path).exists(), "Optimized model not created"
        print("OK: Optimized model created")

        _prepare_optimized_model(optimized_path)
        print("OK: Optimized model loaded and converted")

        rng = np.random.default_rng()
        test_input = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

        original_input_name = _get_model_input_name(original_path)
        optimized_input_name = _get_model_input_name(optimized_path)

        print(f"Running inference on original model (input: {original_input_name})...")
        original_outputs = _run_model(original_path, {original_input_name: test_input})

        print(f"Running inference on optimized model (input: {optimized_input_name})...")
        optimized_outputs = _run_model(optimized_path, {optimized_input_name: test_input})

        original_output_list = list(original_outputs.values())
        optimized_output_list = list(optimized_outputs.values())

        assert len(original_output_list) == len(optimized_output_list), (
            f"Output count mismatch: {len(original_output_list)} vs {len(optimized_output_list)}"
        )

        for i, (orig_out, opt_out) in enumerate(
            zip(original_output_list, optimized_output_list, strict=False)
        ):
            assert orig_out.shape == opt_out.shape, f"Shape mismatch for output {i}"
            assert np.allclose(orig_out, opt_out, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL), (
                f"Output values mismatch for output {i}"
            )

        print("OK: Outputs match between original and optimized models")


def test_conv_bn_fusion() -> None:
    """Test Conv-BatchNorm fusion optimization."""
    print("\nCreating test model for Conv-BN fusion...")
    model = create_test_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        original_path = str(tmpdir_path / "original.onnx")
        optimized_path = str(tmpdir_path / "optimized.onnx")

        onnx.save(model, original_path)

        original_node_count = len(model.graph.node)
        print(f"Original model: {original_node_count} nodes")

        print("Running SlimONNX with Conv-BN fusion...")
        config = OptimizationConfig(
            has_batch_dim=True,
            fuse_conv_bn=True,
            simplify_node_name=True,
        )
        slimonnx = SlimONNX()
        slimonnx.slim(original_path, optimized_path, config=config)

        _prepare_optimized_model(optimized_path)

        optimized_model = onnx.load(optimized_path)
        optimized_node_count = len(optimized_model.graph.node)
        print(f"Optimized model: {optimized_node_count} nodes")

        assert optimized_node_count < original_node_count, (
            f"Node count not reduced: {original_node_count} -> {optimized_node_count}"
        )
        print(f"OK: Node count reduced from {original_node_count} to {optimized_node_count}")

        rng = np.random.default_rng()
        test_input = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

        original_input_name = _get_model_input_name(original_path)
        optimized_input_name = _get_model_input_name(optimized_path)

        print(f"Running inference on original model (input: {original_input_name})...")
        original_outputs = _run_model(original_path, {original_input_name: test_input})

        print(f"Running inference on optimized model (input: {optimized_input_name})...")
        optimized_outputs = _run_model(optimized_path, {optimized_input_name: test_input})

        original_output_list = list(original_outputs.values())
        optimized_output_list = list(optimized_outputs.values())

        assert len(original_output_list) == len(optimized_output_list), "Output count mismatch"

        for i, (orig_out, opt_out) in enumerate(
            zip(original_output_list, optimized_output_list, strict=False)
        ):
            assert orig_out.shape == opt_out.shape, f"Shape mismatch for output {i}"

            max_diff = np.max(np.abs(orig_out - opt_out))
            print(f"Max difference for output {i}: {max_diff:.2e}")

            assert np.allclose(orig_out, opt_out, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL), (
                f"Output values mismatch for output {i}, max diff: {max_diff}"
            )

        print("OK: Outputs match after Conv-BN fusion")
