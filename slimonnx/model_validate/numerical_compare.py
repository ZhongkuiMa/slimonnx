"""Numerical comparison utilities for ONNX models."""

__docformat__ = "restructuredtext"
__all__ = [
    "generate_inputs_from_bounds",
    "run_onnx_inference",
    "compare_model_outputs",
]

import numpy as np
import onnx
import onnxruntime as ort


def generate_inputs_from_bounds(
    input_bounds: tuple[list[float], list[float]],
    input_shape: tuple[int, ...],
    num_samples: int = 5,
) -> list[dict[str, np.ndarray]]:
    """Generate test inputs from lower/upper bounds.

    :param input_bounds: Tuple of (lower_bounds, upper_bounds) lists
    :param input_shape: Shape of input tensor
    :param num_samples: Number of input samples to generate
    :return: List of input dictionaries
    """
    lower_bounds, upper_bounds = input_bounds

    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("Lower and upper bounds must have same length")

    inputs = []
    for _ in range(num_samples):
        sample = np.random.uniform(
            low=lower_bounds, high=upper_bounds, size=input_shape
        ).astype(np.float32)

        inputs.append({"input": sample})

    return inputs


def run_onnx_inference(
    model_path: str,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run ONNX Runtime inference on model.

    :param model_path: Path to ONNX model
    :param inputs: Dictionary of input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    if len(input_names) == 1 and "input" in inputs:
        feed_dict = {input_names[0]: inputs["input"]}
    else:
        feed_dict = inputs

    outputs = session.run(output_names, feed_dict)

    return {name: output for name, output in zip(output_names, outputs)}


def compare_model_outputs(
    model1_path: str,
    model2_path: str,
    input_bounds: tuple[list[float], list[float]] | None = None,
    test_inputs: list[dict[str, np.ndarray]] | None = None,
    test_data_path: str | None = None,
    num_samples: int = 5,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict:
    """Compare outputs of two ONNX models.

    :param model1_path: Path to first model (e.g., original)
    :param model2_path: Path to second model (e.g., optimized)
    :param input_bounds: Tuple of (lower_bounds, upper_bounds) for input generation
    :param test_inputs: Pre-generated inputs (if bounds not provided)
    :param test_data_path: Path to test data file (.npy, .npz)
    :param num_samples: Number of random samples if generating inputs
    :param rtol: Relative tolerance for comparison
    :param atol: Absolute tolerance for comparison
    :return: Comparison report dictionary
    """
    if test_inputs is None:
        if test_data_path is not None:
            try:
                try:
                    from slimonnx.test.utils import load_test_data_from_file

                    test_inputs = load_test_data_from_file(test_data_path)
                except ImportError:
                    if test_data_path.endswith(".npy"):
                        data = np.load(test_data_path)
                        if len(data.shape) == 1:
                            test_inputs = [{"input": data}]
                        else:
                            test_inputs = [
                                {"input": data[i]}
                                for i in range(min(num_samples, data.shape[0]))
                            ]
                    elif test_data_path.endswith(".npz"):
                        data = np.load(test_data_path)
                        if "inputs" in data:
                            inputs_data = data["inputs"]
                            test_inputs = [
                                {"input": inputs_data[i]}
                                for i in range(min(num_samples, inputs_data.shape[0]))
                            ]
                        elif "X" in data:
                            X = data["X"]
                            test_inputs = [
                                {"input": X[i]}
                                for i in range(min(num_samples, X.shape[0]))
                            ]
                    else:
                        raise ValueError(
                            f"Unsupported test data format: {test_data_path}"
                        )

                if test_inputs is not None and len(test_inputs) > num_samples:
                    test_inputs = test_inputs[:num_samples]

            except (IOError, OSError, ValueError, IndexError, KeyError) as error:
                print(f"Failed to load test data: {error}")
                test_data_path = None
                test_inputs = None

        if test_inputs is None and input_bounds is not None:
            model = onnx.load(model1_path)
            input_shape = tuple(
                d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim
            )

            test_inputs = generate_inputs_from_bounds(
                input_bounds, input_shape, num_samples
            )

        if test_inputs is None:
            from .. import utils

            model = onnx.load(model1_path)
            test_inputs = utils.generate_random_inputs(model, num_samples)

    session1 = ort.InferenceSession(model1_path, providers=["CPUExecutionProvider"])
    session2 = ort.InferenceSession(model2_path, providers=["CPUExecutionProvider"])

    input_names1 = [inp.name for inp in session1.get_inputs()]
    output_names1 = [out.name for out in session1.get_outputs()]
    input_names2 = [inp.name for inp in session2.get_inputs()]
    output_names2 = [out.name for out in session2.get_outputs()]

    passed = 0
    failed = 0
    max_diff = 0.0
    mismatches = []

    for i, inputs in enumerate(test_inputs):
        outputs1 = run_onnx_inference(model1_path, inputs)

        if input_names1 != input_names2:
            inputs2 = {}
            for name1, name2 in zip(input_names1, input_names2):
                if name1 in inputs:
                    inputs2[name2] = inputs[name1]
                elif "input" in inputs and len(input_names2) == 1:
                    inputs2[name2] = inputs["input"]
        else:
            inputs2 = inputs

        outputs2 = run_onnx_inference(model2_path, inputs2)

        match = True
        for idx, (out1_name, out2_name) in enumerate(zip(output_names1, output_names2)):
            if out1_name not in outputs1 or out2_name not in outputs2:
                match = False
                mismatches.append(
                    f"Test {i}: Output name mismatch ({out1_name} vs {out2_name})"
                )
                continue

            out1 = outputs1[out1_name]
            out2 = outputs2[out2_name]

            if not np.allclose(out1, out2, rtol=rtol, atol=atol):
                match = False
                diff = np.max(np.abs(out1 - out2))
                max_diff = max(max_diff, diff)
                mismatches.append(
                    f"Test {i}: Output {idx} differs (max_diff={diff:.2e})"
                )

        if match:
            passed += 1
        else:
            failed += 1

    return {
        "all_match": failed == 0,
        "num_tests": len(test_inputs),
        "passed": passed,
        "failed": failed,
        "max_diff": max_diff,
        "mismatches": mismatches,
    }
